import torch


def generate_outputs(self, model, X, y, requires_grad, get_accuracy):

    #my_batch_size = len(X)

    input_ids = self.tokenizer(X, return_tensors="pt", padding="longest", truncation=True).to(self.device)

    answer_ids = self.tokenizer(y, return_tensors="pt", padding="longest", truncation=True).input_ids.to(self.device)
    #may need to select the 0 index here ^^^

    answer_ids = answer_ids[:, -1]

    print(input_ids['input_ids'].shape)
    print(answer_ids.shape)

    torch.cuda.empty_cache()
    
    if requires_grad == False:
        with torch.no_grad():
            logits = model(**input_ids).logits
    else:
        logits = model(**input_ids).logits

    print(logits)

    #shift_logits = logits[..., :-1, :].contiguous()
    #shift_labels = answer_ids[..., :-1, :].contiguous()


    loss = torch.nn.CrossEntropyLoss()(logits[:, -1, :], answer_ids)

    #answer_ids = shift_labels[:, -1]

    if get_accuracy == True:
        top_tokens = torch.topk(logits[:, -1, :], 10, dim=-1).indices  # shape: (batch_size, top_k)

        top1_predictions = top_tokens[:, 0]
        top1_correct = (top1_predictions == answer_ids).sum().item()
        top10_correct = sum([answer_ids[j].item() in top_tokens[j].tolist() for j in range(len(answer_ids))])

        return loss.item(), top1_correct, top10_correct

    else: 

        return loss

    