import torch


def generate_outputs(self, model, X, y, requires_grad, get_accuracy):

    my_batch_size = len(X)

    input_ids = self.tokenizer(X, return_tensors="pt", padding="longest").to(self.device)

    mask_token_id = self.tokenizer.convert_tokens_to_ids('<mask>')
    mask_ids = (input_ids["input_ids"] == mask_token_id).float().argmax(dim=1)

    answers = [gold_answer if gold_answer.startswith(" ") else f" {gold_answer}" for gold_answer in y]

    print(self.tokenizer(answers)["input_ids"])

    answer_ids = [self.tokenizer(answer)["input_ids"][1] for answer in answers]

    answer_ids = torch.LongTensor(answer_ids).unsqueeze(1).to(self.device)

    print(answer_ids)

    answer_ids = answer_ids[:,0]

    if requires_grad == False:
        with torch.no_grad():
            logits = model(**input_ids).logits
    else:
        logits = model(**input_ids).logits

    print(answer_ids)

    mask_ids = mask_ids.view(my_batch_size, 1, 1).expand([my_batch_size, 1, logits.shape[2]])
    masked_logits = torch.gather(logits, index=mask_ids, dim=1)
    loss = torch.nn.CrossEntropyLoss()(masked_logits[:,-1,:], answer_ids)

    

    if get_accuracy == True:
        top_tokens = torch.topk(masked_logits, 10, dim=-1).indices  # shape: (batch_size, top_k)

        top1_predictions = top_tokens[:, 0]
        print(top1_predictions)

        top1_correct = (top_tokens[:,0,0] == answer_ids).sum().item()
        top10_correct = sum([answer_ids[j].item() in top_tokens[j].tolist() for j in range(len(answer_ids))])

        return loss.item(), top1_correct, top10_correct

    else: 

        return loss
    