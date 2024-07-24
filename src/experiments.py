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


    def get_token_ids(self, X, y):

        # Initialize lists to store tokenized questions and numerical answers
        input_ids_list = []
        gold_answer_token_ids_list = []
        
        with torch.no_grad():
            for question, answer in zip(X, y):
                # Tokenize the question
                inputs = self.tokenizer(question, return_tensors="pt", padding='max_length', truncation=True).to(self.device)
                input_ids_list.append(inputs.input_ids)

                # Tokenize the answer
                gold_answer_token_ids = self.tokenizer(answer)["input_ids"]
                
                #answer_len = len(gold_answer_token_ids)
                #assert answer_len == 1, f"For this special case, we assume the answer has 1 token. Found {gold_answer_token_ids}."
                
                gold_answer_token_id = int(gold_answer_token_ids[0])
                gold_answer_token_ids_list.append(gold_answer_token_id)


        input_ids_tensor = torch.cat(input_ids_list, dim=0).to(self.device)
        gold_answer_token_ids_tensor = torch.tensor(gold_answer_token_ids_list).to(self.device)

        return input_ids_tensor, gold_answer_token_ids_tensor
    


    def evaluate(self, model, X, y):

        model.eval() # set model to evaluation mode
        
        input_ids_tensor, gold_answer_token_ids_tensor = self.get_token_ids(X, y)

        outputs = model(input_ids_tensor)
        logits = outputs.logits

        loss = self.loss_fn(logits[:, -1, :], gold_answer_token_ids_tensor)
        

        # Calculate accuracy
        predictions = logits[:, -1, :].argmax(dim=-1)
        correct_predictions = (predictions == gold_answer_token_ids_tensor).sum().item()
        accuracy = correct_predictions / len(X)

        model.train() # return model to train mode

        return loss, accuracy



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




    def fine_tune(self):

        torch.cuda.empty_cache()

        self.load_dataset()

        self.loss_fn = torch.nn.CrossEntropyLoss()

        original_loss, original_accuracy = self.evaluate(self.original_model, self.X, self.y)

        print(f"Original Loss:{original_loss.item()}")
        print(f"Original Accuracy:{original_accuracy}")

        parameters = self.get_parameters()

        for name, param in parameters:
            print(name)
            self.edited_model = deepcopy(self.original_model)


            self.edited_model, self.trainable_parameters, norm, relative_error = self.intervention(name, param)


            optimizer = torch.optim.Adam(self.trainable_parameters, lr=self.args.learning_rate)
            


            print(self.trainable_parameters[0].data[:5, :5])


            for epoch in range(self.args.num_epochs):
                X_train_shuffled, y_train_shuffled = shuffle(self.X_train, self.y_train)

                for i in tqdm(range(0, len(self.X_train), self.args.batch_size)):
                    my_batch_size = min(self.args.batch_size, len(self.X_train) - i)

                    X_batch = X_train_shuffled[i: i + my_batch_size]
                    y_batch = y_train_shuffled[i: i + my_batch_size]

                    batch_loss, _ = self.evaluate(self.edited_model, X_batch, y_batch)

                    print(f"Batch Loss:{batch_loss.item()}")

                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    print(self.trainable_parameters[0].data[:5, :5])
                    
                    torch.cuda.empty_cache()

                epoch_loss, epoch_accuracy = self.evaluate(self.edited_model, self.X_val, self.y_val)

                best_loss = 0

                # Print some stuff
                print(f"Epoch: {epoch}, Epoch Loss: {epoch_loss}, Epoch Accuracy {epoch_accuracy}, Epoch Perplexity: {torch.exp(torch.tensor(epoch_loss)).item()}, Original Loss: {original_loss}, Best Loss: {best_loss}")

                # Write something to preserve the best model and return to this at the end

            final_loss, final_accuracy = self.evaluate(self.edited_model, self.X, self.y)

            
                    





    