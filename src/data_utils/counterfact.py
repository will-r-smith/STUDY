import pickle

def load_dataset(self):

    with open("data/counterfact", "rb") as f:
        data = pickle.load(f)

    eval_set_length = int(self.args.prop_data * len(data))

    data = data[:eval_set_length]

    num_dp = len(data)
    dataset = []

    X, y = [], []

    for i in range(num_dp):

        question = data[i]["question"]
        answer = data[i]["gold-answer"]
        assert answer.startswith(" "), f"Found answer that doesn't start with space ${answer}$"
        dataset.append((question, answer))

    for i in range(len(dataset)):
        question, answer = dataset[i]

        if question.endswith(" "):
            question = f"{question}<mask>."
        else:
            question = f"{question} <mask>."


        X.append(question)
        y.append(answer)

    return X, y