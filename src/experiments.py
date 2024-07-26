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
                'mask_token': '<mask>'
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




    def fine_tune(self):

        torch.cuda.empty_cache()

        self.load_dataset()

        self.loss_fn = torch.nn.CrossEntropyLoss()

        original_loss, original_accuracy = self.evaluate(self.original_model, self.X, self.y)

        #print(f"Original Loss:{original_loss.item()}")
        #print(f"Original Accuracy:{original_accuracy}")

        parameters = self.get_parameters()

        for name, param in parameters:
            #print(name)
            self.edited_model = deepcopy(self.original_model)

            self.edited_model, self.trainable_parameters, norm, relative_error = self.intervention(name, param)

            # Ensure parameters have requires_grad=True
            for param in self.trainable_parameters:
                param.requires_grad = True

            optimizer = torch.optim.Adam(self.trainable_parameters, lr=self.args.learning_rate)

            # print(self.trainable_parameters[0].data[:5, :5])

            for epoch in range(self.args.num_epochs):
                X_train_shuffled, y_train_shuffled = shuffle(self.X_train, self.y_train)

                for i in tqdm(range(0, len(self.X_train), self.args.batch_size)):
                    my_batch_size = min(self.args.batch_size, len(self.X_train) - i)

                    torch.cuda.empty_cache()
                    X_batch = X_train_shuffled[i: i + my_batch_size]
                    y_batch = y_train_shuffled[i: i + my_batch_size]

                    input_ids_tensor, attention_mask_tensor, gold_answer_token_ids_tensor = self.get_token_ids(X_batch, y_batch)

                    torch.cuda.empty_cache()
                    outputs = self.edited_model(input_ids_tensor, attention_mask=attention_mask_tensor)
                    logits = outputs.logits
                    torch.cuda.empty_cache()
                    batch_loss = self.loss_fn(logits[:, -1, :], gold_answer_token_ids_tensor)

                    #print(f"Batch Loss:{batch_loss.item()}")

                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    #print(self.trainable_parameters[0].data[:5, :5])

                epoch_loss, epoch_accuracy = self.evaluate(self.edited_model, self.X_val, self.y_val)

                best_loss = 0

                # Print some stuff
                #print(f"Epoch: {epoch}, Epoch Loss: {epoch_loss}, Epoch Accuracy {epoch_accuracy}, Epoch Perplexity: {torch.exp(torch.tensor(epoch_loss)).item()}, Original Loss: {original_loss}, Best Loss: {best_loss}")

                # Write something to preserve the best model and return to this at the end

            final_loss, final_accuracy = self.evaluate(self.edited_model, self.X, self.y)
                
                    


    def get_token_ids(self, X, y):

        input_ids = self.tokenizer(X, return_tensors="pt", padding="longest").to(self.device)

        mask_token_id = self.tokenizer.convert_tokens_to_ids('<mask>')
        mask_ids = (input_ids["input_ids"] == mask_token_id).float().argmax(dim=1)

        answers = [gold_answer if gold_answer.startswith(" ") else f" {gold_answer}" for gold_answer in y]

        answer_ids = []
        for answer in answers:
            id = self.tokenizer(answer)["input_ids"]
            answer_ids.append(id[1])

        answer_ids = torch.LongTensor(answer_ids).unsqueeze(1).to(self.device)  # batch x 1

        return input_ids, mask_ids, answer_ids



    def evaluate(self, model, X, y):
        model.eval()  # set model to evaluation mode

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        input_ids, mask_ids, answer_ids = self.get_token_ids(X, y)

        batch_size = 8
        total_loss = 0.0

        for i in tqdm(range(0, len(X), 8)):
            my_batch_size = min(batch_size, len(X) - i)
            batch_x = X[i: i + my_batch_size]
            batch_y = y[i: i + my_batch_size]
            input_ids, mask_ids, answer_ids = self.get_token_ids(batch_x, batch_y)

            with torch.no_grad():
                logits = model(**input_ids).logits
                logprob = torch.log_softmax(logits, dim=2)

            print(logprob.shape)
            print(logits.shape)

            vocab_size = logprob.shape[2]
            print(vocab_size)
            mask_ids = mask_ids.view(my_batch_size, 1, 1)
            mask_ids = mask_ids.expand([my_batch_size, 1, vocab_size])

            predicted_logprob = torch.gather(logprob, index=mask_ids, dim=1)     # batch size x 1 x vocab_size
            predicted_logprob = predicted_logprob[:, 0, :]                             # batch x vocab_size

            # Generate top-k tokens
            sorted_logprob, sorted_indices = torch.sort(predicted_logprob, descending=True)    # both are batch x vocab_size
            sorted_logprob = sorted_logprob[:, :10].detach().cpu().numpy()                    # batch x k
            sorted_indices = sorted_indices[:, :10].detach().cpu().numpy()                    # batch x k

            # Compute top-k accuracy
            batch_top_10_tokens = [
                [self.tokenizer.decode(sorted_indices[j, l]).lower().strip() for l in range(10)]
                for j in range(my_batch_size)
            ]
            """
            for b in range(8):
                print(batch_x[b])
                print(batch_top_10_tokens[b])
                print(batch_y[b])"""



        """
        for idx, (question, answer) in enumerate(zip(X, y)):
            input_ids_tensor, attention_mask_tensor, gold_answer_token_ids_tensor = self.get_token_ids([question], [answer])

            with torch.no_grad():
                torch.cuda.empty_cache()
                outputs = model(input_ids_tensor, attention_mask=attention_mask_tensor)
                
                logits = outputs.logits


                #print(logits[0, 63, :10])

                # Align logits with gold_answer_token_ids_tensor shape
                logits = logits[:, -1, :]

                # Calculate loss over the entire sequence
                loss = self.loss_fn(logits, gold_answer_token_ids_tensor)
                total_loss += loss.item()
                #print(loss)

                predictions = logits.argmax(dim=-1)
                #print(predictions)

                # Decode the predictions and gold answers
                predicted_text = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
                gold_answer_text = self.tokenizer.decode(gold_answer_token_ids_tensor[0], skip_special_tokens=True)
                
                
                if idx < 5:  # Print only for the first 20 datapoints
                    print(f"Question: {question}")
                    print(f"Predicted Answer: {predicted_text}")
                    print(f"Gold Answer: {gold_answer_text}")
                    print(f"Original Answer: {answer}\n")

                correct_predictions += (predictions == gold_answer_token_ids_tensor).sum().item()
                total_predictions += gold_answer_token_ids_tensor.size(-1)

            avg_loss = total_loss / len(X)
            accuracy = correct_predictions / len(X)"""

        model.train()  # return model to train mode

        avg_loss = 0
        accuracy = 0

        return avg_loss, accuracy
    



"""



    def get_token_ids(self, X, y):
        input_ids_list = []
        attention_mask_list = []
        gold_answer_token_ids_list = []



        with torch.no_grad():
            for question, answer in zip(X, y):
                #inputs = self.tokenizer(question, return_tensors="pt", padding='max_length', truncation=True, max_length=64).to(self.device)
                inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
                #stripped_answer = answer.strip()
                

                input_ids_list.append(inputs.input_ids)
                attention_mask_list.append(inputs.attention_mask)

                gold_answer_token_ids = self.tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids
                #print(gold_answer_token_ids)
                #gold_answer_token_id = int(gold_answer_token_ids[1])
                gold_answer_token_ids_list.append(gold_answer_token_ids)

        print(gold_answer_token_ids_list)
        input_ids_tensor = torch.cat(input_ids_list, dim=0).to(self.device)
        attention_mask_tensor = torch.cat(attention_mask_list, dim=0).to(self.device)
        gold_answer_token_ids_tensor = torch.tensor(gold_answer_token_ids_list).to(self.device)

        return input_ids_tensor, attention_mask_tensor, gold_answer_token_ids_tensor



    def evaluate(self, model, X, y):
        model.eval()  # set model to evaluation mode

        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for i in range(10):
            print(f"Q:{X[i]}A:{y[i]}")

        for idx, (question, answer) in enumerate(zip(X, y)):
            input_ids_tensor, attention_mask_tensor, gold_answer_token_ids_tensor = self.get_token_ids([question], [answer])

            with torch.no_grad():
                torch.cuda.empty_cache()
                outputs = model(input_ids_tensor, attention_mask=attention_mask_tensor)
                
                logits = outputs.logits


                #print(logits[0, 63, :10])

                # Align logits with gold_answer_token_ids_tensor shape
                logits = logits[:, -1, :]

                # Calculate loss over the entire sequence
                loss = self.loss_fn(logits, gold_answer_token_ids_tensor)
                total_loss += loss.item()
                print(loss)

                predictions = logits.argmax(dim=-1)
                print(predictions)

                # Decode the predictions and gold answers
                predicted_text = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
                gold_answer_text = self.tokenizer.decode(gold_answer_token_ids_tensor[0], skip_special_tokens=True)

                if idx < 5:  # Print only for the first 20 datapoints
                    print(f"Question: {question}")
                    print(f"Predicted Answer: {predicted_text}")
                    print(f"Gold Answer: {gold_answer_text}")
                    print(f"Original Answer: {answer}\n")

                correct_predictions += (predictions == gold_answer_token_ids_tensor).sum().item()
                total_predictions += gold_answer_token_ids_tensor.size(-1)

            avg_loss = total_loss / len(X)
            accuracy = correct_predictions / len(X)

        model.train()  # return model to train mode

        return avg_loss, accuracy
    

    """