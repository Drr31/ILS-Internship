# 📘 Week 2: Python Testing and Testing Frameworks
Explore Python’s testing ecosystem and understand how to apply different testing frameworks to components of the CoVerNet pipeline, ensuring code correctness and robustness.

## Key Activities:
-Reviewed core Python testing tools:

-unittest (Python’s built-in framework)

-pytest (concise syntax and powerful fixtures)

-doctest (tests embedded in docstrings)


## Studied testing concepts like:

-Unit testing

-Edge case handling

-Assertions

-Test-driven snippets

##🔍 Practical Testing Examples Applied to CoVerNet Code
During this phase, I applied testing logic to isolated components of the pipeline to understand how to validate their behavior under different inputs.


```python
🧪 1. Unittest (Python built-in)
Used for classic, class-based testing.

Example 1: Using unittest to Verify Synthetic Data Generation
```


```python
import numpy as np
import pandas as pd

def generate_synthetic_data(sample_size):
    def generate_class_data(x1_range, x2_range, label, n_samples):
        x1 = np.random.uniform(x1_range[0], x1_range[1], n_samples)
        x2 = np.random.uniform(x2_range[0], x2_range[1], n_samples)
        labels = [label] * n_samples
        return pd.DataFrame({'x1': x1, 'x2': x2, 'label': labels})

    class_defs = [
        ([1, 2], [1, 2], 'A'),
        ([1, 2], [2, 3], 'B'),
        ([2, 3], [1, 2], 'C'),
        ([2, 3], [2, 3], 'D')
    ]

    sets = {'train': [], 'val': [], 'test': []}

    for split in ['train', 'val', 'test']:
        split_data = []
        n = sample_size.get(split, 0)
        for x1_range, x2_range, label in class_defs:
            df = generate_class_data(x1_range, x2_range, label, n)
            split_data.append(df)
        full_split = pd.concat(split_data, ignore_index=True).sample(frac=1).reset_index(drop=True)
        sets[split] = full_split

    def split_xy(df):
        x = df[['x1', 'x2']].to_numpy()
        y = df['label'].to_numpy()
        return x, y

    train_x, train_y = split_xy(sets['train'])
    val_x, val_y = split_xy(sets['val'])
    test_x, test_y = split_xy(sets['test'])

    return train_x, train_y, val_x, val_y, test_x, test_y

```


```python
import unittest

class TestDataShape(unittest.TestCase):
    def test_data_shape(self):
        train_x, train_y, _, _, _, _ = generate_synthetic_data({'train': 10, 'val': 0, 'test': 0})
        self.assertEqual(train_x.shape[0], 40)  # 10 samples x 4 classes
        self.assertEqual(train_x.shape[1], 2)
        self.assertEqual(train_y.shape[0], 40)

# Run test in Jupyter
unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestDataShape))

```

    .
    ----------------------------------------------------------------------
    Ran 1 test in 0.010s
    
    OK





    <unittest.runner.TextTestResult run=1 errors=0 failures=0>



2. Pytest (popular 3rd-party tool)
More flexible, supports lightweight functions without classes.



```python
# Example with Clustring :
from sklearn.cluster import KMeans
import numpy as np 

def test_kmeans_clusters():
    X = np.random.rand(100, 4)
    model = KMeans(n_clusters=3, random_state=42).fit(X)
    assert len(set(model.labels_)) == 3

# Run manually in Jupyter
test_kmeans_clusters()
print("Pytest-style function ran without assertion error.")

```

    Pytest-style function ran without assertion error.


🧪 3. Doctest (inline testing inside docstrings)
Great for verifying expected output in documentation.


```python
# Example: PEC generation
import doctest
def equivalence_classes_generation(clusters):
    equivalence_classes = {}
    # Default mapping 0->'A', 1->'B', etc.
    class_mapping = {i: chr(65 + i) for i in range(len(np.unique(clusters)))}
    for cluster in np.unique(clusters):
        class_label = class_mapping[cluster]
        equivalence_classes[class_label] = np.where(clusters == cluster)[0]
    return equivalence_classes
    
def simple_equivalence_example():
    """
    >>> clusters = np.array([0, 1, 0, 2, 1, 0, 2, 1])
    >>> ec = equivalence_classes_generation(clusters)
    >>> sorted(ec.keys())
    ['A', 'B', 'C']
    """
    pass


doctest.run_docstring_examples(simple_equivalence_example, globals(),verbose=True)
```

    Finding tests in NoName
    Trying:
        clusters = np.array([0, 1, 0, 2, 1, 0, 2, 1])
    Expecting nothing
    ok
    Trying:
        ec = equivalence_classes_generation(clusters)
    Expecting nothing
    ok
    Trying:
        sorted(ec.keys())
    Expecting:
        ['A', 'B', 'C']
    ok


### Summary
This week, I gained practical experience with three main Python testing frameworks — unittest, pytest, and doctest. I applied them to various CoVerNet pipeline components, from synthetic data generation to clustering and equivalence class formation. This exploration helped me understand how to validate code correctness and prepare for more robust implementations and verifications in upcoming weeks.
