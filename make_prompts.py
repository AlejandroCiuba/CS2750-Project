from langchain.prompts import PromptTemplate
from pathlib import Path

import pandas as pd

def make_prompt(template: Path | str, **kwargs) -> str:

    text = """"""
    try:
        with open(file=template, mode='r') as src:
            text = src.read(-1)
    except Exception as e:
        raise e
    
    return PromptTemplate.from_template(template=text).format(**kwargs)


def iter_prompt(template: Path | str, data: pd.DataFrame):

    text = """"""
    try:
        with open(file=template, mode='r') as src:
            text = src.read(-1)
    except Exception as e:
        raise e
    
    prompt = PromptTemplate.from_template(template=text)

    for row in data.to_dict(orient='records'):
        yield prompt.format(**row)


if __name__ == "__main__":

    def full_sentence(row: pd.Series) -> list[str]:
        before, after, pleonasm = row.before, row.after, row.consensus
        if pleonasm == 'neither' or pleonasm == 'both':
            return f"{before.strip()} {after.strip()}"
        else:
            return f"{before.strip()} {pleonasm.strip()} {after.strip()}"

    df = pd.read_json("", lines=True)
    df = df.sample(10, random_state=50)

    df['review'] = df.apply(full_sentence, axis=1)
    df['task'] = df.apply(lambda x: "Identify the pleonasm in the customer review.", axis=1)
    df['format'] = df.apply(lambda x: "{\"pleonasm\": \"<WORD>\"}", axis=1)

    for text in iter_prompt(template=Path(""), data=df):
        print(text)