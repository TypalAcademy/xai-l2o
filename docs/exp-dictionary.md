---
hide:
  - toc
---

# Implicit Dictionary Learning

This is where we overview the model for the toy experiment.

!!! tip "Full Tutorial"
    See the tutorial page for a Jupyter notebook using this model. :fontawesome-regular-face-laugh-wink: 

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
::: src.models.ImpDictModel 
    options:
      show_root_heading: false
      show_source: false  
      members: None

<br>

## Apply Optimization Step    
::: src.models.ImpDictModel._apply_T 
    options:
      heading_level: 2
      show_root_heading: false
      show_source: true            

<br>

## Get Convergence Criteria
::: src.models.ImpDictModel._get_conv_crit 
    options:
      show_root_heading: false
      show_source: true       

<br>

## Forward  
::: src.models.ImpDictModel.forward 
    options:
      show_root_heading: false
      show_source: true        
