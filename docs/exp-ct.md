---
hide:
  - toc
---

# Implicit Deep Learning for CT

This is where we overview the model for the CT experiments

<!-- !!! tip "Full Tutorial"
    See the tutorial page for a Jupyter notebook using this model. :fontawesome-regular-face-laugh-wink: -->

<!-- ## Sample Code Usage

Insert sample code

``` py title="bubble_sort.py" linenums="1" hl_lines="2 3"
def bubble_sort(items):
    for i in range(len(items)):
        for j in range(len(items) - 1 - i):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
``` -->

## Model Overview
::: src.models.CT_L2O_Model
    options:
      show_root_heading: false
      show_source: false  
      members: None

<br>

## Apply Optimization Step    
::: src.models.CT_L2O_Model._apply_T
    options:
      heading_level: 2
      show_root_heading: false
      show_source: true            

<br>

## Get Convergence Criteria
::: src.models.CT_L2O_Model._get_conv_crit
    options:
      show_root_heading: false
      show_source: true       

<br>

## Forward  
::: src.models.CT_L2O_Model.forward
    options:
      show_root_heading: false
      show_source: true        


## Downloading Dataset

The dataset can be downloaded [here](https://drive.google.com/drive/folders/1Z0A3c-D4dnrhlXM8cpgC1b7Ltyu0wpgQ)
