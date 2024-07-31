
import torch

def generate_outputs(self, model_eval, X, y, requires_grad, get_accuracy):

    bs = len(X)

    input_ids = self.tokenizer(X, return_tensors="pt", padding="longest").to(self.device)

    mask_token_id = self.tokenizer.convert_tokens_to_ids('<mask>')
    mask_ids = (input_ids["input_ids"] == mask_token_id).float().argmax(dim=1)

    answers = [gold_answer if gold_answer.startswith(" ") else f" {gold_answer}" for gold_answer in y]


    answer_ids = [self.tokenizer(answer)["input_ids"][1] for answer in answers]

    answer_ids = torch.LongTensor(answer_ids).unsqueeze(1).to(self.device)


    answer_ids = answer_ids[:,0]

    torch.cuda.empty_cache()

    if requires_grad == False:
        with torch.no_grad():
            logits = model_eval(**input_ids).logits
    else:
        logits = model_eval(**input_ids).logits


    mask_ids = mask_ids.view(bs, 1, 1).expand([bs, 1, logits.shape[2]])
    masked_logits = torch.gather(logits, index=mask_ids, dim=1)
    loss = torch.nn.CrossEntropyLoss()(masked_logits[:,-1,:], answer_ids)


    if get_accuracy == True:
        top_tokens = torch.topk(masked_logits, 10, dim=-1).indices  # shape: (batch_size, top_k)

        top1_correct = (top_tokens[:,0,0] == answer_ids).sum().item()
        top10_correct = sum([answer_ids[j].item() in top_tokens[j,:,0].tolist() for j in range(len(answer_ids))])
        
        top1_words = [self.tokenizer.decode([token]) for token in top_tokens]
        top10_words = [[self.tokenizer.decode([token]) for token in tokens] for tokens in top_tokens]

        
        for idx, tokens in enumerate(top10_words):
            print(y[idx])
            print(f'Top 10 tokens for masked position {idx} in batch: {tokens}')
        

        return loss.item(), top1_correct, top10_correct, top1_words, top10_words

    else: 

        return loss
    