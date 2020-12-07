## CNN Wafer Inspection Results

Graphs showing the results obtained using a learning rate of 0.0001, the adam optimizer, and 128 neurons in a fully connected layer, for:
1. Categorical Classification for the entire dataset
![](results%20categorical_entire_dataset.png)
2. Categorical Classification for the entire dataset (to distinguish between the different failure modes and not-failed), and additionally using data augmentation - horizontal and vertical flips
![](results%20categorical_entire_dataset_augmented.png)
3. Categorical Classification just for the failed wafers (to distinguish between the different failure modes only), and using data augmentation
![](results%20categorical_only_failed_augmented.png)
4. Binary Classification for the entire dataset to tell whether a wafer has failed or not, also using data augmentation
![](results%20binary_entire_dataset_augmented.png)
5. Binary Classification for equal parts of failed and non-failed wafers to tell whether a wafer has failed or not, once again using data augmentation
![](results%20binary_equal_datasets_augmented.png)
