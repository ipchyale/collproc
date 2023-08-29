import os
import pandas as pd
import re

listtypes = (list,tuple,set)

def dirwalk(dir: str, omit=None, require=None) -> list:
    if omit is None:
        omit = []
    if require is None:
        require = []

    if not isinstance(omit,listtypes):
        omit = [omit]
    if not isinstance(require,listtypes):
        require = [require]

    files = []
    for root, _, filenames in os.walk(dir):
        for f in filenames:
            files.append(os.path.join(root,f))

    for o in omit:
        files = [f for f in files if o not in f]
    for r in require:
        files = [f for f in files if r in f]

    return files

def findhead(excel_path: str, threshold=0.7) -> int:
    # Read the Excel file with no header
    df = pd.read_excel(excel_path, header=None)

    # Ignore any column with all null values
    df = df.loc[:, df.isnull().mean() < 1]

    # Iterate over the rows
    for i, row in df.iterrows():
        # Check the fraction of non-empty cells in the row
        if row.dropna().count() / len(row) >= threshold:
            return i  # This is likely the header row

    return None  # No header detected

def ftriage(f: str) -> pd.DataFrame:
    ext = os.path.splitext(f)[1]
    if ext == '.csv':
        return pd.read_csv(f)
    elif ext == '.xls' or ext == '.xlsx':
        header = findhead(f)
        df = pd.read_excel(f,header=header)
        df = df.loc[:, df.isnull().mean() < 1] # Drop any column with all null values
        return df
    else:
        raise ValueError('Filetype not supported.')
    
def findcat(df: pd.DataFrame) -> pd.Series:
    """
    Finds the column with catalog numbers in a dataframe. Because a catalog 
    number might mimic a numeric value, we eliminate from consideration any
    column whose values lie within the physical measurement ranges we typically
    see in our data. We then return the column with the highest number of
    unique values.

    This is not a failsafe solution and may need modification for edge cases
    we haven't yet encountered.
    """

    # edge case: exports from the ref collection database use two columns for catalog numbers, so we concatenate them
    if 'Catalog Number' in df.columns and 'Secondary Catalog Number' in df.columns:
        df['Catalog Number'] = df['Catalog Number'].fillna('')
        df['Secondary Catalog Number'] = df['Secondary Catalog Number'].fillna('')
        df['Catalog Number'] = df['Catalog Number'].apply(str)
        df['Secondary Catalog Number'] = df['Secondary Catalog Number'].apply(str)
        df['catalog'] = df['Catalog Number'] + df['Secondary Catalog Number']
        
        return 'catalog'
    
    # Iterate over columns and exclude those which match the specific criteria
    potential_id_columns = []
    for col in df.columns:

        # Exclude datetime columns from consideration
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue  # skip to the next column
       
        # Check if the column values can be converted to float
        try:
            col_as_float = df[col].astype(float)
        except ValueError:
            potential_id_columns.append(col)
            continue  # skip to the next column

        # Filter based on your described criteria
        if not (
            (0 <= col_as_float).all() and (col_as_float <= 115).all() or 
            (-8 <= col_as_float).all() and (col_as_float <= 52).all() or 
            (0 <= col_as_float).all() and (col_as_float <= 0.8).all() or 
            (0 <= col_as_float).all() and (col_as_float <= 0.5).all()
        ):
            potential_id_columns.append(col)

    # If no potential id columns remain, return None
    if not potential_id_columns:
        return None

    # Return name of column with the highest number of unique values
    return df[potential_id_columns].nunique().idxmax()

def findvals(df: pd.DataFrame, dimension='thickness', threshold=0.8) -> list:
    # Define value ranges for each dimension
    dimension_ranges = {
        'thickness': (0, 0.8),
        'gloss': (0, 115),
        'roughness': (0, 0.5),
        'color': (-8, 52)
    }

    if dimension not in dimension_ranges:
        raise ValueError(f"Unknown dimension '{dimension}'. Options are: {', '.join(dimension_ranges.keys())}")

    # Get the range for the selected dimension
    lower_bound, upper_bound = dimension_ranges[dimension]

    potential_cols = []
    for col in df.columns:

        # Exclude datetime columns from consideration
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue  # skip to the next column

        valid_count = 0
        for val in df[col]:
            try:
                float_val = float(val)
                if dimension=='thickness':
                    float_val = abs(float_val) # depth gauge reports negative values
                if lower_bound <= float_val <= upper_bound:
                    valid_count += 1
            except (ValueError, TypeError):
                continue

        if valid_count / len(df[col]) >= threshold:
            potential_cols.append(col)

    return potential_cols

def fproc(f: str, fname=True, dimension='thickness', threshold=0.8) -> pd.DataFrame:

    df = ftriage(f)
    catcol = findcat(df)
    valcols = findvals(df,dimension,threshold)

    if not catcol:
        print(f'No valid catalog column found in {f}')
        return None

    if not valcols:
        print(f'No valid value columns found in {f}')
        return None
    
    # rename catalog column
    df.rename(columns={catcol:'catalog'},inplace=True)

    # some valcol cells will be strings we cannot convert to float, so we convert them to NaN
    df[valcols] = df[valcols].apply(pd.to_numeric,errors='coerce')

    # convert all values to positive
    df[valcols] = df[valcols].abs()

    # create column where each cell is a list of all non-null values from the valcols
    df[dimension] = df[valcols].apply(lambda x: x.dropna().tolist(),axis=1)

    # if list is empty, drop the row
    df = df[df[dimension].map(len) > 0]        

    # remove any rows where the catalog number is null
    df = df[df['catalog'].notnull()]

    # select only catalog and dimension columns
    df = df[['catalog',dimension]]

    # create a string version of the dimension column
    df[dimension+'_str'] = df[dimension].apply(str)

    # drop duplicates based on all columns except `dimension` (because Python cannot hash lists)
    df = df.drop_duplicates(subset=df.columns.difference([dimension]))
        
    if fname:
        df['fname'] = os.path.basename(f)

    df = df.reset_index(drop=True)
    
    return df

def extract_catalog_number(s):
    match = re.search(r'(\d+[a-z]*)', s)
    if match:
        return match.group(1)
    return None

def extract_two_letter_code(s):
    # Look for a pattern of a letter followed by either a letter or a digit.
    match = re.search(r'([a-zA-Z][a-zA-Z\d])', s)
    if match:
        return match.group(1)
    return None

def extract_single_digit(s):
    # Look for a pattern of a single digit either preceded by an underscore or surrounded by underscores.
    match = re.search(r'(?<=_)\d(?=_|$)', s)
    if match:
        return match.group(0)
    return None

def is_consecutive(series: pd.Series) -> bool:
    # Take the difference between consecutive items
    differences = series.diff().dropna()  # drop the first item which will be NaN
    
    # Check if all differences are equal to 1
    return (differences == 1).all()