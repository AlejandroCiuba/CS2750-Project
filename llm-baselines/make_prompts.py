from gensim.models import KeyedVectors
from langchain.prompts import PromptTemplate
from nltk import word_tokenize
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
            pleonasm = f"\"{row['first']}\", \"{row['second']}\""
        else:
            pleonasm = f"\"{row['consensus']}\""

        if 'examples' in row.keys():
            yield prompt.format(**row), row['review'], row['examples'], pleonasm
        else:
            yield prompt.format(**row), row['review'], pleonasm


def full_sentence(row: pd.Series) -> list[str]:
    before, after, first, second = row.before, row.after, row['first'], row.second
    return f"{before} {first} {second} {after}"


# Cannot just get sentences from the 'review' column as we need to get both parts if there are two pleonasms.
def few_shot(row: pd.Series, sample_pool: pd.DataFrame, examples: int = 4, linguistic_feature = None, lf_kwargs = {}):

    samples = sample_pool.sample(examples)

    preamble = "In the sentence:"
    fewshot = """"""
    for sample in samples.itertuples():

        before, after, first, second, pleonasm = sample.before, sample.after, sample.first, sample.second, sample.consensus
        sentence = f"{before} {first} {second} {after}"

        if pleonasm == 'neither':
            fewshot += f"'''{preamble} \"{sentence}\", there are no pleonasms.'''\n"
        elif pleonasm == 'both':
            fewshot += f"'''{preamble} \"{sentence}\", the pleonasms are {first} and {second}.'''\n"
        else:
            fewshot += f"'''{preamble} \"{sentence}\", the pleonasm is {pleonasm}.'''\n"

        if linguistic_feature is not None:
            fewshot += f"And {linguistic_feature(sample, **lf_kwargs)}\n"

    return fewshot.strip()


def highest_cosine_similarity_between_neighbors(row: pd.Series, embeddings: KeyedVectors) -> pd.Series:
    """
    `embeddings`: KeyedVector object from gensim
    """
    sent = word_tokenize(row.review)

    top = ("", "", 0.0)
    for ind in range(len(sent)):
        if ind != len(sent) - 1:
            try:
                sim = embeddings.similarity(sent[ind], sent[ind + 1])
                if sim > top[2]:
                    top = (sent[ind], sent[ind + 1], sim)
            except:
                continue

    return f"the two neighboring words with the highest cosine similarity: \"{top[0]}\" and \"{top[1]}\" at {top[2]:.3f}"


if __name__ == "__main__":
    
    df = pd.read_json("../data/SPC-FOLD/SPC.json", lines=True)
    df['review'] = df.apply(full_sentence, axis=1)

    vecs = KeyedVectors.load_word2vec_format("PATH", binary=True)

    rest = df[df['fold'] != 0]
    test = df[df['fold'] == 0].sample(10, random_state=50)

    test['task'] = df.apply(lambda x: "Identify the pleonasm in the customer review.", axis=1)
    test['format'] = df.apply(lambda x: "{\"pleonasm\": \"<WORD>\"}", axis=1)
    test['examples'] = df.apply(few_shot, axis=1, sample_pool=rest, examples=4, 
                                linguistic_feature=highest_cosine_similarity_between_neighbors, lf_kwargs={"embeddings": vecs})
    test['linguistic_feature'] = df.apply(highest_cosine_similarity_between_neighbors, axis=1, embeddings=vecs)

    for text in iter_prompt(template=Path("prompts/few-shot-linguistic.txt"), data=test):
        print(text)