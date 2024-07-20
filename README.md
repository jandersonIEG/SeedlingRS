# SeedlingRS
Using UAV remote sensing, particularly multispectral imagery, to detect planted seedlings and monitor for mortality and vigor. The imagery was collected in late summer, the year of the tree planting, and used geolocated (marked) seedlings from 30 100m2 plots as training data for identification of other seedlings accross the whole site (11 ha). The workflow does involve working a geomatics platform, and either moving back and forth, or running the python codes here through an imbedded console (I have mostly used this with QGIS). 
 
The initial workflow (detailed below) requires three tools:
<ul>
  <li>*Cluster_attributing* - This imports clusters of vegetation, and attributes them with size, shape, texture and spectral characteristic.</li>
  <li>*Cluster classification model - Tree_Nontree_RF&XGBoost* - this tool uses random forest and gradient boosting models to make predictions, based on the attributed characteristics above, about whether the vegetation cluster is a seedline or a non-seedling cluster of vegetation. The training data is built using the plots, where all seedlings were marked and attributed, therefore any non-marked clusters were attributed as non-seedlings. </li>
  <li>*Classification model - Tree species_RF_only* - This tool is similar to the above tool, but gradient boosting is dropped due to overfitting. This tool predicts the species of each seedling cluster. The training data are the marked, identified trees from the plots.</li>
</ul>

WorkFlow:
1. Collect multispectral UAV imagery late summer after planting.
2. Mark and ID all seedlings, using GNSS rover, within plots for training data
3. Process imagery, and then manually filter and vectorize areas which are likely vegetation (ei. NDVI>0.2) 
4. Attribute clusters which are to be used as training data (marked seedlings)
5. Attribute clusters with size, shape, texture and spectral characteristics using the *Cluster_attributing* tool
6. Create predictions for whether a cluster is a seedling, a non-seedling plant or other (such as wooden stakes or flagging tape), using the *Cluster classification model - Tree_Nontree_RF&XGBoost* tool
7. Perform manual QA/QC on the results
8. Create predictions for seedling species using the *Classification model - Tree species_RF_only* tool 
9. Perform manual QA/QC on the results

