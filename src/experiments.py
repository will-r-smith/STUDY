import pandas as pd
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import importlib
import os

from transformers import AutoTokenizer

from src.matrix_utils import norms, do_lr
from src.eval_utils.linguistics import classify_words

from accelerate import Accelerator

#from google.colab import files

import nltk

import spacy

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')



class Experiment:

    def __init__(self, args, config):

        #nltk.download('averaged_perceptron_tagger')
        #nltk.download('punkt')
        #nltk.download('stopwords')

        # Load spaCy model
        #nlp = spacy.load("en_core_web_sm")

        

        torch.cuda.empty_cache()

        self.args = args
        self.config = config

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.args.verbose > 0:
            print(f"Device: {self.device}")
        
        self.load_model()


    def load_model(self):

        print(f"\nModel: {self.config[self.args.model]['name']}")

        if self.args.verbose > 0:
            print("\nLoading model...")

        if self.args.model == "roberta_base":
            from transformers import RobertaForMaskedLM
            llm_name = "roberta-base"
            model = RobertaForMaskedLM.from_pretrained(llm_name, cache_dir='./cache')
            
        elif self.args.model == "pythia160m":
            from transformers import AutoModelForCausalLM
            llm_name = "EleutherAI/pythia-160m-deduped"
            model = AutoModelForCausalLM.from_pretrained(llm_name, cache_dir='./cache')
        
        elif self.args.model == "gptj":
            from transformers import GPTJForCausalLM
            llm_name = "EleutherAI/gpt-j-6B"
            model = GPTJForCausalLM.from_pretrained(
                llm_name,
                revision="float16",
                torch_dtype=torch.float16,
                cache_dir='./cache',
            ) 

        elif self.args.model == "roberta_large":
            from transformers import RobertaForMaskedLM
            llm_name = "roberta-large"
            model = RobertaForMaskedLM.from_pretrained(llm_name, cache_dir='./cache')

        elif self.args.model == "pythia410m":
            from transformers import AutoModelForCausalLM
            llm_name = "EleutherAI/pythia-410m-deduped"
            model = AutoModelForCausalLM.from_pretrained(llm_name, cache_dir='./cache')


        self.llm_name = llm_name   

        self.original_model = model

        self.original_model.to(self.device)

        self.edited_model = model
        self.edited_model.to(self.device)

        if self.args.verbose > 0:
            print("Model loaded.")

        self.accelerator = Accelerator()
        self.edited_model = self.accelerator.prepare(self.edited_model)

        #for name, param in self.edited_model.named_parameters():
            #print(name)

            


    def load_dataset(self):

        print(f"\nDataset: {self.args.dataset}")

        if self.args.verbose > 0:
            print("Loading dataset...")


        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)

        if self.args.model in ["pythia160m", "gptj", "pythia410m"]:
            # Add padding and mask token if they don't exist
            self.tokenizer.add_special_tokens({
                'pad_token': self.tokenizer.eos_token,
                #'mask_token': '<mask>'
            })
            self.original_model.resize_token_embeddings(len(self.tokenizer))

        module = importlib.import_module(f"src.data_utils.{self.args.dataset}")
        get_dataset = getattr(module, 'get_dataset')

        self.X, self.y = get_dataset(self, self.config[self.args.model]['type'])

        print("loaded dataset")

        if self.args.test == "fine_tune":
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42, shuffle=True)

        self.dataset_size = len(self.X)

        if self.args.verbose > 0:
            print("Dataset loaded.")
            print(f"Size: {self.dataset_size}")

            

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
            model, approx_mat, parameters, S = do_lr(self.edited_model, name, original_mat_tensor.type(torch.float32), (1 - self.args.rate))


        diff_norm, relative_error = norms(original_mat_tensor.type(torch.float32), approx_mat)


        return model, parameters, diff_norm, relative_error, S



    def intervene(self):

        parameters = self.get_parameters()

        for name, param in parameters:
            print(name)

            self.edited_model, _ , norm, relative_error, S = self.intervention(name, param)

            #loss, accuracy = self.evaluate()
            results = {}

            results["parameter"] = name
            results["rate"] = self.args.rate
            results["SVs"] = str(S.tolist())

            self.terminate_and_save(results)



    def evaluate(self, model, X, y):
        model.eval()

        top1_all = []
        top10_all = []

        total_loss = 0.0
        total_top1_correct = 0
        total_top10_correct = 0


        for i in tqdm(range(0, len(X), self.args.batch_size)):

            my_batch_size = min(self.args.batch_size, len(X) - i)

            batch_x = X[i: i + my_batch_size]
            batch_y = y[i: i + my_batch_size]

            batch_loss, top1_correct, top10_correct, top1_words, top10_words = self.generate_outputs(self, model, batch_x, batch_y, False, True)

            top1_all.extend(top1_words)
            top10_all.extend([word for sublist in top10_words for word in sublist])
            
            total_loss += batch_loss
            total_top1_correct += top1_correct
            total_top10_correct += top10_correct
            
        # Compute average loss for the batch
        average_loss = total_loss / len(X)

        top1_frequencies = classify_words(top1_all)
        top10_frequencies = classify_words(top10_all)

        # Compute accuracies
        top1_accuracy = total_top1_correct / len(X)
        top10_accuracy = total_top10_correct / len(X)

        return average_loss, top1_accuracy, top10_accuracy, top1_frequencies, top10_frequencies
    



    def fine_tune(self):
        
        if self.args.model in ["roberta_base", "roberta_large"]:
            loc = "src.eval_utils.masked"
        else: 
            loc = "src.eval_utils.causal"

        module = importlib.import_module(loc)
        self.generate_outputs = getattr(module, 'generate_outputs')

        torch.cuda.empty_cache()

        self.load_dataset()

        self.loss_fn = torch.nn.CrossEntropyLoss()

        original_loss, original_top1_accuracy, original_top10_accuracy, original_top1_words, original_top10_words = self.evaluate(self.original_model, self.X_val, self.y_val)
    

        original_results = {'original_loss': original_loss, 
                            'original_top1_accuracy': original_top1_accuracy, 
                            'original_top10_accuracy': original_top10_accuracy,
                            'original_top1_words': str(original_top1_words),
                            'original_top10_words': str(original_top10_words)}


        if self.args.verbose > 0:
            print(f"Original Loss: {original_loss}")
        if self.args.verbose > 1:
            print(f"Original Top-1 Accuracy {original_top1_accuracy}")
        if self.args.verbose > 2:
            print(f"Original Top-10 Accuracy {original_top10_accuracy}")

        parameters = self.get_parameters()

        for name, param in parameters:

            results = original_results

            results["parameter"] = name
            results["dataset_len"] = self.dataset_size
            results["rate"] = self.args.rate
            results["learning_rate"] = self.args.learning_rate
            results["es"] = self.args.early_stopping
            
            param.requires_grad = False

            print(f"\nPerforming invervention on: {name}")
            print(f"  {self.config['Arguments']['intervention']['values'][self.args.intervention]}\n")

            #del self.edited_model
            torch.cuda.empty_cache()

            self.edited_model = deepcopy(self.original_model)

            self.edited_model.to(self.device)


            torch.cuda.empty_cache()

            self.edited_model, self.trainable_parameters, norm, relative_error, S = self.intervention(name, param)
            
            results["norm"] = norm.item()
            results["relative_error"] = relative_error

            self.edited_model.to(self.device)

            edited_loss, edited_top1_accuracy, edited_top10_accuracy, edited_top1_words, edited_top10_words = self.evaluate(self.edited_model, self.X_val, self.y_val)


            results["edited_loss"] = edited_loss
            results["edited_top1_accuracy"] = edited_top1_accuracy
            results["edited_top10_accuracy"] = edited_top10_accuracy
            results["edited_top1_categories"] = edited_top1_words
            results["edited_top10_categories"] = edited_top10_words

            torch.cuda.empty_cache()

            if self.args.verbose > 0:
                print(f"  Edited Loss: {edited_loss}")
            if self.args.verbose > 1:
                print(f"  Edited Top-1 Accuracy {edited_top1_accuracy}")
            if self.args.verbose > 2:
                print(f"  Edited Top-10 Accuracy {edited_top10_accuracy}")

            # Ensure parameters have requires_grad=True
            for param in self.trainable_parameters:
                param.requires_grad = True

            optimizer = torch.optim.Adam(self.trainable_parameters, lr=self.args.learning_rate)
            self.optimizer = self.accelerator.prepare(optimizer)

            if self.args.verbose > 3:
                print(f"            P[0,0]:   {self.trainable_parameters[0].data[0, 0].item()}")


            self.scaler = torch.cuda.amp.GradScaler()
            self.edited_model.train()

            es = 0
            best_loss = np.inf
            epoch_losses = []

            for epoch in range(self.args.num_epochs):

                if self.args.verbose > 0:
                    print(f"  \nEpoch {epoch}\n")

                X_train_shuffled, y_train_shuffled = shuffle(self.X_train, self.y_train)

                self.epoch_train(X_train_shuffled, y_train_shuffled)

                if self.args.verbose > 3:
                    print(self.trainable_parameters[0].data[0, 0].item())


                epoch_loss, epoch_top1_accuracy, epoch_top10_accuracy, _, _ = self.evaluate(self.edited_model, self.X_val, self.y_val)


                if self.args.verbose > 1:
                    print(f"    Epoch {epoch} Loss: {epoch_loss}")
                if self.args.verbose > 1:
                    print(f"    Epoch {epoch} Top-1 Accuracy {epoch_top1_accuracy}")
                if self.args.verbose > 2:
                    print(f"    Epoch {epoch} Top-10 Accuracy {epoch_top10_accuracy}")

                epoch_losses.append(epoch_loss)

                if epoch_loss < best_loss:
                    es = 0
                    best_loss = epoch_loss
                else:
                    es +=1

                if es > self.args.early_stopping:
                    break

            results["epoch_losses"] = str(epoch_losses)

            final_loss, final_top1_accuracy, final_top10_accuracy, final_top1_words, final_top10_words = self.evaluate(self.edited_model, self.X, self.y)


            results["final_loss"] = final_loss
            results["final_top1_accuracy"] = final_top1_accuracy
            results["final_top10_accuracy"] = final_top10_accuracy
            results["finally_top1_categories"] = final_top1_words
            results["final_top10_categories"] = final_top10_words

            print(f"Finished fine-tuning layer: {name}")

            if self.args.verbose > 0:
                print(f"  Final Loss: {final_loss}")
            if self.args.verbose > 1:
                print(f"  Final Top-1 Accuracy {final_top1_accuracy}")
            if self.args.verbose > 2:
                print(f"  Final Top-10 Accuracy {final_top10_accuracy}")

            self.terminate_and_save(results)

                    

    def terminate_and_save(self, results):

        path = f"/content/drive/My Drive/project_results/{self.args.intervention}/{self.args.model}/{self.args.dataset}.csv"
        
        results_df = pd.DataFrame([results])

        # Check if the CSV file exists and is not empty
        if os.path.exists(path):
            try:
                existing_results_df = pd.read_csv(path)
                df = pd.concat([existing_results_df, results_df], ignore_index=True)
            except pd.errors.EmptyDataError:
                df = results_df
        else:
            df = results_df

        df.to_csv(path, index=False)




    def simple_fine_tune(self):
        if self.args.model in ["roberta_base", "roberta_large"]:
            loc = "src.eval_utils.masked"
        else:
            loc = "src.eval_utils.causal"

        module = importlib.import_module(loc)
        self.generate_outputs = getattr(module, 'generate_outputs')

        torch.cuda.empty_cache()

        self.load_dataset()

        self.loss_fn = torch.nn.CrossEntropyLoss()

        original_loss, original_top1_accuracy, original_top10_accuracy, original_top1_words, original_top10_words = self.evaluate(self.original_model, self.X_val, self.y_val)

        original_results = {
            'original_loss': original_loss, 
            'original_top1_accuracy': original_top1_accuracy, 
            'original_top10_accuracy': original_top10_accuracy,
            'original_top1_words': str(original_top1_words),
            'original_top10_words': str(original_top10_words)
        }

        if self.args.verbose > 0:
            print(f"Original Loss: {original_loss}")
        if self.args.verbose > 1:
            print(f"Original Top-1 Accuracy {original_top1_accuracy}")
        if self.args.verbose > 2:
            print(f"Original Top-10 Accuracy {original_top10_accuracy}")

        parameters = self.get_parameters()

        for name, param in parameters:
            results = original_results.copy()

            results["parameter"] = name
            results["dataset_len"] = self.dataset_size
            results["learning_rate"] = self.args.learning_rate
            results["es"] = self.args.early_stopping
                
            torch.cuda.empty_cache()

            # Deep copy the original model and move it to the device
            self.edited_model = deepcopy(self.original_model).to(self.device)

            # Set all parameters to not require gradients
            for p in self.edited_model.parameters():
                p.requires_grad = False

            # Find the specific parameter in the edited model and set requires_grad=True
            trainable_param = None
            for n, p in self.edited_model.named_parameters():
                if n == name:
                    p.requires_grad = True
                    trainable_param = p
                    break

            if trainable_param is None:
                raise ValueError(f"Parameter {name} not found in the model")

            optimizer = torch.optim.Adam([trainable_param], lr=self.args.learning_rate)
            self.optimizer = self.accelerator.prepare(optimizer)

            self.scaler = torch.cuda.amp.GradScaler()
            self.edited_model.train()

            es = 0
            best_loss = float('inf')
            epoch_losses = []

            for epoch in range(self.args.num_epochs):
                if self.args.verbose > 0:
                    print(f"  \nEpoch {epoch}\n")

                X_train_shuffled, y_train_shuffled = shuffle(self.X_train, self.y_train)

                self.epoch_train(X_train_shuffled, y_train_shuffled)

                epoch_loss, epoch_top1_accuracy, epoch_top10_accuracy, _, _ = self.evaluate(self.edited_model, self.X_val, self.y_val)

                if self.args.verbose > 0:
                    print(f"    Epoch {epoch} Loss: {epoch_loss}")
                if self.args.verbose > 1:
                    print(f"    Epoch {epoch} Top-1 Accuracy {epoch_top1_accuracy}")
                if self.args.verbose > 2:
                    print(f"    Epoch {epoch} Top-10 Accuracy {epoch_top10_accuracy}")

                epoch_losses.append(epoch_loss)

                if epoch_loss < best_loss:
                    es = 0
                    best_loss = epoch_loss
                else:
                    es += 1

                if es > self.args.early_stopping:
                    break

            results["epoch_losses"] = str(epoch_losses)

            final_loss, final_top1_accuracy, final_top10_accuracy, final_top1_words, final_top10_words = self.evaluate(self.edited_model, self.X, self.y)

            results["final_loss"] = final_loss
            results["final_top1_accuracy"] = final_top1_accuracy
            results["final_top10_accuracy"] = final_top10_accuracy
            results["finally_top1_categories"] = final_top1_words
            results["final_top10_categories"] = final_top10_words

            print(f"Finished fine-tuning layer: {name}")

            if self.args.verbose > 0:
                print(f"  Final Loss: {final_loss}")
            if self.args.verbose > 1:
                    print(f"  Final Top-1 Accuracy {final_top1_accuracy}")
            if self.args.verbose > 2:
                print(f"  Final Top-10 Accuracy {final_top10_accuracy}")

            self.terminate_and_save(results)


    def epoch_train(self, X_train_shuffled, y_train_shuffled):
        for i in tqdm(range(0, len(self.X_train), self.args.batch_size)):
            my_batch_size = min(self.args.batch_size, len(self.X_train) - i)

            batch_x = X_train_shuffled[i: i + my_batch_size]
            batch_y = y_train_shuffled[i: i + my_batch_size]

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                batch_loss = self.generate_outputs(self, self.edited_model, batch_x, batch_y, True, False)

            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            torch.cuda.empty_cache()

            if self.args.verbose > 3:
                print(f"        Batch loss: {batch_loss}")