import torch


def generate_outputs(self, model, X, y, requires_grad, get_accuracy):

    #my_batch_size = len(X)

    y = [gold_answer if gold_answer.startswith(" ") else f" {gold_answer}" for gold_answer in y]

    input_ids = self.tokenizer(X, return_tensors="pt", padding="longest", truncation=True).to(self.device)

    #answer_ids = self.tokenizer(y, return_tensors="pt").input_ids  #.to(self.device)

    answer_ids = [self.tokenizer(answer)["input_ids"][0] for answer in y]

    answer_ids = torch.LongTensor(answer_ids).unsqueeze(1).to(self.device)


    # Compute the lengths of the original input sequences
    input_lengths = [len(self.tokenizer.encode(x, truncation=True)) for x in X]

    # Extract the last token index for each sequence before padding
    answer_positions = torch.tensor([input_lengths[i] - 1 for i in range(len(X))]).to(self.device)


    answer_ids = answer_ids[:, -1]


    torch.cuda.empty_cache()

    
    if requires_grad == False:
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = model(**input_ids).logits
    else:
        with torch.cuda.amp.autocast():
            logits = model(**input_ids).logits

    answer_ids = answer_ids.to(torch.long)


    answer_logits = logits[torch.arange(logits.size(0)), answer_positions]


    torch.cuda.empty_cache()
    loss = torch.nn.CrossEntropyLoss()(answer_logits, answer_ids)


    if get_accuracy == True:
        top_tokens = torch.topk(answer_logits, 10, dim=-1).indices  # shape: (batch_size, top_k)
        top1_predictions = top_tokens[:, 0]
        top1_correct = (top1_predictions == answer_ids).sum().item()
        top10_correct = sum([answer_ids[j].item() in top_tokens[j].tolist() for j in range(len(answer_ids))])
        
        
        top1_words = [self.tokenizer.decode([token]) for token in top1_predictions]
        top10_words = [[self.tokenizer.decode([token]) for token in tokens] for tokens in top_tokens]

        """
        for idx, tokens in enumerate(top10_words):
            print(y[idx])
            print(f'Top 10 tokens for masked position {idx} in batch: {tokens}')
        """

        return loss.item(), top1_correct, top10_correct, top1_words, top10_words

    else: 

        return loss

    