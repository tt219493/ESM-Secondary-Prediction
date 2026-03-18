import polars as pl
from transformers import AutoTokenizer
from datasets import Dataset


def create_sliding_windows(df : pl.LazyFrame, window_size: int, step_size = None) -> pl.LazyFrame:
  '''
  Given a LazyFrame with columns 
  `id`, `asym_id`, `sequence`, `label`, `index`, `input_ids`, `attention_mask` (not aggregated), 
  returns a LazyFrame with windowed sequences and labels.

  Parameters
  ---
  df: pl.LazyFrame
    LazyFrame to window
  window_size: int
    Size of window 
  step_size: 
    Step size of window (determines window overlap). Defaults to no overlap

  Returns
  ---
  pl.LazyFrame
    LazyFrame with windowed data
  '''

  period = f'{window_size}i'
  every = f'{step_size}i' if step_size else period

  return (df.sort(['id', 'asym_id', 'index'])
            .group_by_dynamic(index_column='index',
                                    period=period,
                                    every=every,
                                    closed='right',
                                    group_by=['id', 'asym_id'])
                  .agg([pl.col('sequence').first(),
                        pl.col('label'),
                        pl.col('input_ids'),
                        pl.col('attention_mask'),
                        pl.col('index').alias('idx_agg')])
                  .with_columns(sequence=pl.col('sequence').str.slice(pl.col('index'), window_size),
                                index=pl.col('idx_agg'))
                  .drop('idx_agg')
                  .select(['id', 'asym_id', 'sequence', 'label', 'index', 'input_ids', 'attention_mask'])
                  .sort(['id', 'asym_id'])
                  )

def create_multiple_windows(df: pl.LazyFrame, window_sizes: list[int], has_overlap: bool = True):
    '''
    Given a LazyFrame with columns 
    `id`, `asym_id`, `sequence`, `label`, `index`, `input_ids`, `attention_mask` (not aggregated)
    and a list of window sizes
    returns a concatenated LazyFrame with multiple windowed sequences and labels.

    Parameters
    ---
    df: pl.LazyFrame
        LazyFrame to window
    window_sizes: list[int]
        List of window sizes
    has_overlap: bool = True 
        True makes step_size = window_size // 2 and False means no overlap between windows (step_size = window_size)

    Returns
    ---
    pl.LazyFrame
        LazyFrame with concatenated windowed data    
    '''

    if has_overlap:
        df_dict = {f'{ws}': create_sliding_windows(df, ws, ws // 2) for ws in window_sizes}
    else:
        df_dict = {f'{ws}': create_sliding_windows(df, ws) for ws in window_sizes}

    temp_df = df.clear().group_by(['id', 'asym_id', 'sequence']
                                  ).agg([
                                        pl.col('label'),
                                        pl.col('index'),
                                        pl.col('input_ids'),
                                        pl.col('attention_mask')
                                        ])
    
    for ws in window_sizes:
        temp_df = pl.concat([temp_df, df_dict[f'{ws}']])

    return temp_df


def create_aggregated_windows(df: pl.LazyFrame, window_sizes: list[int], has_overlap: bool = True):
    '''
    Given a LazyFrame with columns 
    `id`, `asym_id`, `sequence`, `label`, `index`, `input_ids`, `attention_mask` (not aggregated)
    and a list of window sizes,
    groups and aggregates the LazyFrame and concatenates all windowed LazyFrames

    Parameters
    ---
    df: pl.LazyFrame
        LazyFrame to window
    window_sizes: list[int]
        List of window sizes
    has_overlap: bool = True 
        True makes step_size = window_size // 2 and False means no overlap between windows (step_size = window_size)

    Returns
    ---
    pl.LazyFrame
        LazyFrame with aggregated data and concatenated windowed data    
    '''

    temp_df = (df.sort(['id', 'asym_id', 'index'])
                    .group_by(['id', 'asym_id', 'sequence']
                                  ).agg([
                                        pl.col('label'),
                                        pl.col('index'),
                                        pl.col('input_ids'),
                                        pl.col('attention_mask')
                                        ])
    )
    temp_df = pl.concat([temp_df, create_multiple_windows(df, window_sizes, has_overlap)])
    # remove duplicates
    temp_df = temp_df.unique(['id', 'asym_id','sequence', 'label', 'index'])
    return temp_df

def train_val_split(df, n: int, offset: int=0):
    '''
    Deterministic split of LazyFrame using IDs. 

    Parameters
    ---
    df
        Polars LazyFrame
    n: int
        Fraction of validation (example: n=5 is 0.2)
    offset: int=0
        Starting index to gather

    Returns
    ---
    tuple(LazyFrame, LazyFrame)
        Full LazyFrames split as (train_split, test_split)


    '''
    ids = df.select('id').unique('id').sort('id')
    val_ids = ids.gather_every(n=n, offset=offset)

    train_ids = val_ids.clear()
    for i in range(0, n):
        if i != offset:
            train_ids = pl.concat([train_ids, ids.gather_every(n=n, offset=i)])

    train_split = df.join(train_ids, on=['id'], how='inner')
    val_split = df.join(val_ids, on=['id'], how='inner')

    return train_split, val_split


def process_benchmark(df, label_mapping):
    temp_df = (df
                .select(['input', ' dssp8'])
                .with_columns(
                    sequence = pl.col('input').str.split(by=" ").list.join(separator=""),
                    secondary_structure = pl.col(' dssp8').str.split(by=" "),
                )
                .with_columns(
                    length = pl.col('sequence').str.len_chars()
                ).with_columns(
                    index = pl.int_ranges(1, pl.col('length') + 1)
                )
                .drop(['input', ' dssp8', 'length'])
              )
    
    temp_df = ( 
                temp_df
                    .unique(['sequence', 'secondary_structure']) #remove duplicates
                    .explode('secondary_structure', 'index')
                    .with_columns(label = pl.col("secondary_structure").replace_strict(label_mapping))
                    .cast({'label' : pl.Int64})
                    .sort('sequence', 'index')
                    .group_by('sequence').agg('secondary_structure', 'index', 'label')      
                ).join(temp_df.select('sequence'), on='sequence') #add duplicates back

    return temp_df

def tokenize_benchmark(df, label_mapping, tokenizer):    
    def tokenize_and_label(examples):
        tokenized_inputs = tokenizer(examples["sequence"],
                                        return_tensors="pt",
                                        add_special_tokens=False,
                                        padding=False)
        tokenized_inputs["input_ids"] = tokenized_inputs["input_ids"][0]
        tokenized_inputs["attention_mask"] = tokenized_inputs["attention_mask"][0]
        return tokenized_inputs
    
    ds = Dataset.from_polars(process_benchmark(df, label_mapping).collect())
    tokenized_ds = ds.map(tokenize_and_label, batched=False)
    tokenized_df = Dataset.to_polars(tokenized_ds).lazy()
    return tokenized_df

def remove_overlapping_seq(df, df_list):
    temp_df = df
    for test_df in df_list:
        temp_df = (temp_df.join(test_df.select('sequence'), on=['sequence'], how='full')
                          .filter(pl.col('sequence_right').is_null()).drop('sequence_right')
                  )
    return temp_df