import polars as pl

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



