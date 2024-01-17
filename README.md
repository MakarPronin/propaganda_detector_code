# PropagandaDetector
This propaganda detector uses the BERT model to label text samples with the type of propaganda they contain (If they do).

# How to use:

## A) Train the model (requires GPU with CUDA):
1. Download the repository.
2. Run these commands in the console:
    pip install transformers==4.24.0
    pip install datasets==2.7.1
    pip install evaluate==0.3.0
3. Update files in the datasets folder (don't change names and format of the files).
4. Open propagandaDetector.py.
5. Uncomment "# trainModel()" line in propagandaDetector.py.
6. Run propagandaDetector.py.
7. Wait until the operation is complete.


## B) Test the model:
1. Open propagandaDetector.py.
2. Add this line with your text samples:
    print(giveLabel(["Sample1", "Sample2"]))
3. Run propagandaDetector.py.
4. Observe the output of the following format:
     [['someLable1'], ['someLable2']]

[''] label stands for "not propaganda."

### Examples:
**Input**:  "Ukrainian President Vladimir Zelensky has 
            maintained since the start of the conflict that his military will retake all of Ukraine’s former territories, including Crimea. However, his long-promised summer counteroffensive failed to land Ukraine more than a handful of frontline villages and resulted in the loss of over 125,000 troops and 16,000 pieces of heavy equipment, according to the latest figures from the Russian Defense Ministry."\
**Output**: "Loaded_Language"

**Input**:  "Something is wrong with our country.", "The 
            conversation was quite stressful", "I'm happy today."\
**Output**: "Flag-Waving", "", ""

**Input**:  "Eating fresh fruits and vegetables is good for 
            your health"\
**Output**: ""