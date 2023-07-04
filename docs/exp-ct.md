---
hide:
  - toc
---

# Implicit Deep Learning for CT

Herein we overview the model and setup for the CT image reconstruction experiments.

## CT Data

The datasets used in this set of experiments are stored in a publicly accesible Google Drive folder.

<center>
[Download CT Data](https://drive.google.com/drive/folders/1Z0A3c-D4dnrhlXM8cpgC1b7Ltyu0wpgQ){ .md-button .md-button--primary }
</center>

<br>

## CT Model Overview
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
