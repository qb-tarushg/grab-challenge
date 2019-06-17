# Grab Challenge - Computer Vision

How might we automate the process of recognizing the details of the vehicles from images, including make and model?
This is a data science assignment where you are expected to create a data model from a given training dataset.

PROBLEM STATEMENT

Given a dataset of distinct car images, can you automatically recognize the car model and make?

## Pipeline

The steps are detailed under jupyter notebook `Pipeline.ipynb`.

### Evaluate the model

Run code as :

```python
car_net.evaluate(test_loader)
```

### Using Predict Method as below 

```python
test_iter = iter(test_loader)
car_net.predict(test_iter.next()[0], k=3)
```

### Load the model from checkpoint

```python
car_net = CarNet(NUM_CLASSES, require_chkpoint=True, chkpnt_folder=checkpoint_folder)
car_net.load_checkpoint("val_chk.pt")
```

And then you can run the evaluation as mentioned above