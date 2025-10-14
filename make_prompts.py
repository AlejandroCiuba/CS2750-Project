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

        if row['consensus'] == 'neither':
            pleonasm = "\"NONE\""
        elif row['consensus'] == 'both':
            pleonasm = f"\"{row['before'].split(' ')[-1]}\", \"{row['after'].split(' ')[0]}\""
        else:
            pleonasm = f"\"{row['consensus']}\""

        if 'examples' in row.keys():
            yield prompt.format(**row), row['review'], row['examples'], pleonasm
        else:
            yield prompt.format(**row), row['review'], pleonasm


if __name__ == "__main__":

    def full_sentence(row: pd.Series):

        before, after, pleonasm = row.before, row.after, row.consensus
        if pleonasm == 'neither' or pleonasm == 'both':
            return f"{before.strip()} {after.strip()}"
        else:
            return f"{before.strip()} {pleonasm.strip()} {after.strip()}"

    def few_shot(row: pd.Series, sample_pool: pd.DataFrame, examples: int = 4):

        samples = sample_pool.sample(examples)

        preamble = "In the sentence:"
        fewshot = """"""
        for sample in samples.itertuples():

            before, after, pleonasm = sample.before, sample.after, sample.consensus
            if pleonasm == 'neither':
                fewshot += f"'''{preamble} \"{before} {after}\", there are no pleonasms.'''\n"
            elif pleonasm == 'both':
                fewshot += f"'''{preamble} \"{before} {after}\", the pleonasms are {before.split(' ')[-1]} and {after.split(' ')[0]}.'''\n"
            else:
                fewshot += f"'''{preamble} \"{before} {pleonasm} {after}\", the pleonasm is {pleonasm}.'''\n"

        return fewshot.strip()

    df = pd.read_json("data/SPC-FOLD/SPC.json", lines=True)
    df['review'] = df.apply(full_sentence, axis=1)

    rest = df[df['fold'] != 0]
    test = df[df['fold'] == 0].sample(10, random_state=50)

    test['task'] = df.apply(lambda x: "Identify the pleonasm in the customer review.", axis=1)
    test['format'] = df.apply(lambda x: "{\"pleonasm\": \"<WORD>\"}", axis=1)
    test['examples'] = df.apply(few_shot, axis=1, args=(rest, 4))

    for text in iter_prompt(template=Path("prompts/few-shot.txt"), data=test):
        print(text)