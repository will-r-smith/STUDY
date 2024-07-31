from datasets import load_dataset


def get_dataset(self, model_type):

    full_dataset = load_dataset("hotpot_qa", "fullwiki")

    eval_set_length_train = int(self.args.prop_data * len(full_dataset["train"]))
    eval_set_length_val= int(self.args.prop_data * len(full_dataset["validation"]))

    # As the hotpot QA does not have answers for test set, we use the train set
    dataset = []
    for dp in full_dataset["train"][:eval_set_length_train]:
        question = dp["question"].strip()
        answer = dp["answer"].strip()
        dataset.append((question, answer))

    for dp in full_dataset["validation"][:eval_set_length_val]:
        question = dp["question"].strip()
        answer = dp["answer"].strip()
        dataset.append((question, answer))


    X, y = [], []

    for i in range(len(dataset)):
        question, answer = dataset[i]

        if not question.endswith("?") and not question.endswith("."):
            question = f"{question}? The answer is"
        else:
            question = f"{question} The answer is"

        if model_type == "masked":
            if question.endswith(" "):
                question = f"{question}<mask>."
            else:
                question = f"{question} <mask>."

        X.append(question)
        y.append(answer)

    return X, y
