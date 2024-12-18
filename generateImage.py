import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from concurrent.futures import ProcessPoolExecutor

def process_img(smiles, mol_index, filename):

    # Create RDKit molecule object from SMILES string
    mol = Chem.MolFromSmiles(smiles)
    # Generate image if molecule is valid
    if mol is not None:
        img = Draw.MolToImage(mol, size=(224, 224))
        img.save(f'./dataset/tmp/{filename}/raw/{mol_index}.png')  # Save image with index as filename
    else:
        print(f"Could not generate molecule for index {mol_index}, invalid SMILES.")


if __name__ == '__main__':
    # **************************** #
    # Format of data: index,SMILES_stand,label
    # **************************** #

    # Load CSV file
    filename = "The name of the data you need to process."
    file_path = './dataset/tmp/{}.csv'.format(filename)  # Replace with the actual path to your CSV file
    df = pd.read_csv(file_path)
    dd = df.to_dict(orient='records')
    # Loop through the DataFrame and generate images for each SMILES string
    res = []
    total = len(dd)

    with ProcessPoolExecutor(max_workers=20) as executor:
        fs = []
        for index, row in enumerate(dd):
            mol_index = row['index']
            tmpRow = row.copy()
            tmpRow["name"] = "{}.png".format(mol_index)
            # tmpRow["label"] = 0.66
            res.append(tmpRow)
            future = executor.submit(process_img, smiles=row['SMILES_stand'], mol_index=row['index'], filename=filename)
            fs.append(future)
            print("\r{}/{}".format(index+1, total), end='')
        print("")
        for index, future in enumerate(fs):
            future.result()
            print("\r{}/{}".format(index+1, total), end='')
    resDf = pd.DataFrame(res)
    resDf.to_csv("./dataset/tmp/{}/{}.csv".format(filename,filename), index=False)
    print("\nImages have been generated and saved as PNG files.")
