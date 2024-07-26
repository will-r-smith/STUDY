import torch
from copy import deepcopy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import importlib

from transformers import AutoTokenizer

from src.matrix_utils import norms, do_lr, do_mm


class Experiment:

    def __init__(self, args, config):
        torch.cuda.empty_cache()

        self.args = args
        self.config = config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")

        self.load_model()


    def load_model(self):

        if self.args.model == "roberta":
            from transformers import RobertaForMaskedLM
            llm_name = "roberta-base"
            model = RobertaForMaskedLM.from_pretrained(llm_name, cache_dir='./cache')
            
        elif self.args.model == "pythia":
            from transformers import AutoModelForCausalLM
            llm_name = "EleutherAI/pythia-160m-deduped-v0"
            model = AutoModelForCausalLM.from_pretrained(llm_name, cache_dir='./cache')
        
        elif self.args.model == "gptj":
            from transformers import GPTJForCausalLM
            llm_name = "EleutherAI/gpt-j-6B"
            model = GPTJForCausalLM.from_pretrained(
                llm_name,
                revision="float16",
                torch_dtype=torch.float16,
                cache_dir='./cache'
            ) 

        self.llm_name = llm_name   

        self.original_model = model

        self.edited_model = model
        self.edited_model.to(self.device)

        print("loaded model")



    def load_dataset(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)

        if self.args.model in ["pythia", "gptj"]:
            # Add padding and mask token if they don't exist
            self.tokenizer.add_special_tokens({
                'pad_token': self.tokenizer.eos_token,
                'mask_token': '[MASK]'
            })
            self.edited_model.resize_token_embeddings(len(self.tokenizer))

        module = importlib.import_module(f"src.data_utils.{self.args.dataset}")
        load_dataset = getattr(module, 'load_dataset')


        self.X, self.y = load_dataset(self)

        print("loaded dataset")

        if self.args.test == "fine_tune":

            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42, shuffle=True)

        self.dataset_size = len(self.X)


    def get_parameters(self):

        layer_mappings = self.config[self.args.model]["naming_conv"]["layers"]

        if self.args.lname[0] == "all":
            layer_keys = layer_mappings.values()
        else:
            layer_keys = []
            for n in self.args.lname:
                layer_keys.append(layer_mappings[n])

        if self.args.lnum[0] == "all":
            layer_numbers = range(self.config[self.args.model]['num_layers'])
        else: 
            layer_numbers = [int(i) for i in self.args.lnum]

        # Collect names of parameters to modify
        params_to_modify = []

        for name, param in self.original_model.named_parameters():
            for key in layer_keys:
                if key in name:
                    for layer_num in layer_numbers:
                        if f".{layer_num}." in name:
                            params_to_modify.append((name, param))

        return params_to_modify



    def save_results(self):
        pass





    def intervention(self, name, param):
        
        original_mat = param.detach().cpu().numpy()
        original_mat_tensor = deepcopy(param)

        if self.args.intervention == "lr":
            model, approx_mat, parameters = do_lr(self.edited_model, name, original_mat_tensor.type(torch.float32), (1 - self.args.rate))

        elif self.args.intervention == "mm":
            model, approx_mat = do_mm(original_mat)

        diff_norm, relative_error = norms(original_mat_tensor.type(torch.float32), approx_mat)

        return model, parameters, diff_norm, relative_error

        


    def intervene(self):

        self.model_edit = deepcopy(self.original_model)

        parameters = self.get_parameters()

        for name, param in parameters:

            self.edited_model, _ , norm, relative_error = self.intervention(name, param)

            loss, accuracy = self.evaluate()

            #self.save_results()


    def get_token_ids(self, question, answer):
        inputs = self.tokenizer(question, return_tensors="pt", padding='max_length', truncation=True).to(self.device)
        input_ids = inputs.input_ids

        gold_answer_token_ids = self.tokenizer(answer)["input_ids"]
        gold_answer_token_id = int(gold_answer_token_ids[0])

        return input_ids, gold_answer_token_id

    def evaluate(self, model, X, y):
        model.eval()
        
        total_loss = 0.0
        correct_predictions = 0

        for question, answer in zip(X, y):
            input_ids_tensor, gold_answer_token_id = self.get_token_ids(question, answer)
            gold_answer_token_id = torch.tensor([gold_answer_token_id]).to(self.device)

            with torch.no_grad():
                outputs = model(input_ids_tensor)
                logits = outputs.logits

                loss = self.loss_fn(logits[:, -1, :], gold_answer_token_id)
                total_loss += loss.item()

                predictions = logits[:, -1, :].argmax(dim=-1)
                correct_predictions += (predictions == gold_answer_token_id).sum().item()

            torch.cuda.empty_cache()

        avg_loss = total_loss / len(X)
        accuracy = correct_predictions / len(X)

        model.train()

        return avg_loss, accuracy

    def fine_tune(self, X_train, y_train, X_val, y_val, args):
        torch.cuda.empty_cache()
        self.load_dataset()
        self.loss_fn = torch.nn.CrossEntropyLoss()

        print(torch.cuda.memory_summary(device=None, abbreviated=False))

        original_loss, original_accuracy = self.evaluate(self.model, self.X, self.y)

        print(f"Original Loss: {original_loss}")
        print(f"Original Accuracy: {original_accuracy}")

        parameters = self.get_parameters()

        scaler = GradScaler()  # For mixed precision

        for name, param in parameters:
            print(name)
            self.edited_model = deepcopy(self.model)
            self.edited_model, self.trainable_parameters, norm, relative_error = self.intervention(name, param)

            optimizer = torch.optim.Adam(self.trainable_parameters, lr=args.learning_rate)

            print(torch.cuda.memory_summary(device=None, abbreviated=False))

            for epoch in range(args.num_epochs):
                X_train_shuffled, y_train_shuffled = shuffle(self.X_train, self.y_train)

                optimizer.zero_grad()

                for i in tqdm(range(0, len(self.X_train), args.batch_size)):
                    my_batch_size = min(args.batch_size, len(self.X_train) - i)

                    X_batch = X_train_shuffled[i: i + my_batch_size]
                    y_batch = y_train_shuffled[i: i + my_batch_size]

                    batch_loss = 0.0

                    for question, answer in zip(X_batch, y_batch):
                        torch.cuda.empty_cache()

                        input_ids_tensor, gold_answer_token_id = self.get_token_ids(question, answer)
                        gold_answer_token_id = torch.tensor([gold_answer_token_id]).to(self.device)

                        with autocast():  # Mixed precision
                            outputs = self.edited_model(input_ids_tensor)
                            logits = outputs.logits
                            loss = self.loss_fn(logits[:, -1, :], gold_answer_token_id)

                        batch_loss += loss

                    batch_loss = batch_loss / args.batch_size
                    print(f"Batch Loss: {batch_loss.item()}")

                    scaler.scale(batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    torch.cuda.empty_cache()

                epoch_loss, epoch_accuracy = self.evaluate(self.edited_model, self.X_val, self.y_val)

                best_loss = 0

                print(f"Epoch: {epoch}, Epoch Loss: {epoch_loss}, Epoch Accuracy {epoch_accuracy}, "
                      f"Epoch Perplexity: {torch.exp(torch.tensor(epoch_loss)).item()}, Original Loss: {original_loss}, Best Loss: {best_loss}")

            final_loss, final_accuracy = self.evaluate(self.edited_model, self.X, self.y)
