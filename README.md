# cost-sensitive-calibration

This package implements a basic version of binary sensitive calibration based in this Medium
[article](https://medium.com/mercadolibre-datablog/cost-sensitive-classification-in-fraud-prevention-263170d8fcfe).

```bash
pip install git+https://github.com/Tokukawa/cost-sensitive-calibration.git
```

So let's say you have a binary classifier that must be calibrated based on a utility matrix.
How can I use this package? Here an actual example:

```python
from cost_sensitive_calibration.calibrate import BinaryCalibration, AcceptReviewReject
import numpy as np

EXAMPLES = 1000
preds = np.random.uniform(size=EXAMPLES)
labels = (preds > np.random.rand(EXAMPLES)) * 1
```

`preds` contain the model predictions and `labels` the true labels. Now let's say our utility matrix
is like this:

<table>
  <tr>
    <th>Tables</th>
    <th>True Positive</th>
    <th>True Negative</th>
  </tr>
  <tr>
    <td>model positive</td>
    <td>0.0</td>
    <td>-0.1</td>
  </tr>
  <tr>
    <td>model negative</td>
    <td>-1.0</td>
    <td>+0.5</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

The utility matrix must expressed in per dollar return. So -1 means -100% returns 0.5 means +50% return and so on.
Than you can use ROC based optimization like this in order to find the optimal threshold:

```python
utility_matrix = {'tp': 0., 'fp': -0.1, 'tn': .5, 'fn': -1.}
caliber = BinaryCalibration(utility_matrix)
threshold, max_utility = caliber.calibrate(labels, preds, plot_roc=False)
print("Optimal Threshold:{} \nMax Utility: {}".format(threshold, max_utility))
```
```bash
>Optimal Threshold:0.316255844096 
>Max Utility: 0.0975
```
In case you want to use an Accept-Review-Reject approach you can use the class `AcceptReviewReject`. Example

```python
from cost_sensitive_calibration.calibrate import AcceptReviewReject
two_thresholds_caliber = AcceptReviewReject(utility_matrix={'PR': 0., 'NR': 0., 'PM': -0.1, 'NM': 0.025 , 'PA': -1, 'NA': 0.025}, steps=1000)
lower_threshold, higher_threshold, utility_per_dollar = two_thresholds_caliber.calibrate(labels, preds)
```

where PR, NR, PM, NM, PA, NA means:

            PR -> Positive Rejected
            NR -> Negative Rejected
            PM -> Positive to Manual Review
            NM -> Negative to Manual Review
            PA -> Positive Accepted
            NA -> Negative Accepted




Or you can use a bayesian approach to take an action without a threshold:

```python
multiple_options_utility_matrix = np.array([[0., -0.1], [-1., .5]])
binary_bayesian_classifier = BinaryBayesianMinimumRisk(multiple_options_utility_matrix)
```
```python
my_pred = 0.12345
action = {0: 'Take action 1', 1: 'Take action 2'}
print(action[binary_bayesian_classifier.predict(my_pred)])
```

```bash
>Take action 2
```

The class `BinaryBayesianMinimumRisk` can be initialized with more than two possible options.
