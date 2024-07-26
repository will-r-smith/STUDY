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




    def fine_tune(self):

        torch.cuda.empty_cache()

        self.load_dataset()

        self.loss_fn = torch.nn.CrossEntropyLoss()

        original_loss, original_top1_accuracy, original_top10_accuracy = self.evaluate(self.original_model, self.X_val, self.y_val)

        print(f"Original Loss: {original_loss}, Original Top-1 Accuracy {original_top1_accuracy}, Original Top-1 Accuracy {original_top10_accuracy}")


        parameters = self.get_parameters()

        for name, param in parameters:
            self.edited_model = deepcopy(self.original_model)

            self.edited_model, self.trainable_parameters, norm, relative_error = self.intervention(name, param)

            edited_loss, edited_top1_accuracy, edited_top10_accuracy = self.evaluate(self.edited_model, self.X_val, self.y_val)

            print(f"Edited Model Loss: {edited_loss}, Edited Model  Top-1 Accuracy {edited_top1_accuracy}, Edited Model  Top-1 Accuracy {edited_top10_accuracy}")

            # Ensure parameters have requires_grad=True
            for param in self.trainable_parameters:
                param.requires_grad = True

            optimizer = torch.optim.Adam(self.trainable_parameters, lr=self.args.learning_rate)

            #print(self.trainable_parameters[0].data[:2, :2])

            for epoch in range(self.args.num_epochs):
                X_train_shuffled, y_train_shuffled = shuffle(self.X_train, self.y_train)

                for i in tqdm(range(0, len(self.X_train), self.args.batch_size)):

                    my_batch_size = min(self.args.batch_size, len(self.X_train) - i)

                    batch_x = X_train_shuffled[i: i + my_batch_size]
                    batch_y = y_train_shuffled[i: i + my_batch_size]
                    
                    input_ids, mask_ids, answer_ids = self.get_token_ids(batch_x, batch_y)

                    torch.cuda.empty_cache()
                    logits = self.edited_model(**input_ids).logits

                    mask_ids = mask_ids.view(my_batch_size, 1, 1).expand([my_batch_size, 1, logits.shape[2]])
                    masked_logits = torch.gather(logits, index=mask_ids, dim=1)

                    batch_loss = torch.nn.CrossEntropyLoss()(masked_logits[:,-1,:], answer_ids)
                    #print(batch_loss)

                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()

                    torch.cuda.empty_cache()

                    
                #print(self.trainable_parameters[0].data[:2, :2])
                epoch_loss, epoch_top1_accuracy, epoch_top10_accuracy = self.evaluate(self.edited_model, self.X_val, self.y_val)


                print(f"Epoch: {epoch}, Epoch Loss: {epoch_loss}, Epoch Top-1 Accuracy {epoch_top1_accuracy}, Epoch Top-1 Accuracy {epoch_top10_accuracy}, Original Loss: {original_loss}")

                #print(f"Epoch: {epoch}, Epoch Loss: {epoch_loss}, Epoch Accuracy {epoch_accuracy}, Epoch Perplexity: {torch.exp(torch.tensor(epoch_loss)).item()}, Original Loss: {original_loss}, Best Loss: {best_loss}")

                # Write something to preserve the best model and return to this at the end

            final_loss, final_top1_accuracy, final_top10_accuracy = self.evaluate(self.edited_model, self.X, self.y)

            print(final_loss)
            print(final_top1_accuracy)
            print(final_top10_accuracy)
                
                    


    def get_token_ids(self, X, y):

        input_ids = self.tokenizer(X, return_tensors="pt", padding="longest").to(self.device)

        mask_token_id = self.tokenizer.convert_tokens_to_ids('<mask>')
        mask_ids = (input_ids["input_ids"] == mask_token_id).float().argmax(dim=1)

        answers = [gold_answer if gold_answer.startswith(" ") else f" {gold_answer}" for gold_answer in y]

        answer_ids = [self.tokenizer(answer)["input_ids"][1] for answer in answers]

        answer_ids = torch.LongTensor(answer_ids).unsqueeze(1).to(self.device)

        return input_ids, mask_ids, answer_ids[:,0]



    def evaluate(self, model, X, y):
        model.eval()  # set model to evaluation mode

        total_loss = 0.0
        total_top1_correct = 0
        total_top10_correct = 0

        input_ids, mask_ids, answer_ids = self.get_token_ids(X, y)

        batch_size = 32


        for i in tqdm(range(0, len(X), batch_size)):
            my_batch_size = min(batch_size, len(X) - i)
            batch_x = X[i: i + my_batch_size]
            batch_y = y[i: i + my_batch_size]
            input_ids, mask_ids, answer_ids = self.get_token_ids(batch_x, batch_y)

            with torch.no_grad():
                logits = model(**input_ids).logits

            mask_ids = mask_ids.view(my_batch_size, 1, 1).expand([my_batch_size, 1, logits.shape[2]])
            masked_logits = torch.gather(logits, index=mask_ids, dim=1)

            loss = torch.nn.CrossEntropyLoss()(masked_logits[:,-1,:], answer_ids)
            total_loss += loss.item()

            top_tokens = torch.topk(masked_logits, 10, dim=-1).indices  # shape: (num_masked_tokens, top_k)

            total_top1_correct += (top_tokens[:,0,0] == answer_ids).sum().item()
            total_top10_correct += sum([answer_ids[j].item() in top_tokens[j,0,:].tolist() for j in range(len(answer_ids))])

            torch.cuda.empty_cache()
            
            #decoded_top_tokens = [[self.tokenizer.decode(token) for token in tokens] for tokens in top_tokens]
            
            """
            # Print or store the top 10 decoded tokens
            for idx, tokens in enumerate(decoded_top_tokens):
                print(batch_x[idx])
                print(f"Answer: {self.tokenizer.decode(answer_ids[idx,0])}")
                print(f'Top 10 tokens for masked position {idx} in batch: {tokens}')
                print(top10_correct)

            """

        # Compute average loss for the batch
        average_loss = total_loss / (len(X) / batch_size)

        # Compute accuracies
        top1_accuracy = total_top1_correct / len(X)
        top10_accuracy = total_top10_correct / len(X)

        # Print or log the final metrics for the batch if needed
        #print(f'Final loss for the batch: {average_loss}')
        #print(f'Top-1 accuracy for the batch: {top1_accuracy}')
        #print(f'Top-10 accuracy for the batch: {top10_accuracy}')

        model.train()  # return model to train mode

        return average_loss, top1_accuracy, top10_accuracy
    



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


"""