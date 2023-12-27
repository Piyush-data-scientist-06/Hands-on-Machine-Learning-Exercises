# Hands-on-Machine-Learning-Exercises
Performed the following exercises and gained hands-on experience in machine learning libraries and algorithms such as Decision trees, Random Forests, and Support Vector Machine (SVM). Implemented the concept of tuning the model (Hyperparameter tuning) and voting classifiers. The exercises done are as follows: <br><br>
**1. Train and fine-tune a Decision Tree for the moon dataset by following these steps:** <br>
a) Use make_moons(n_samples=10000, noise=0.4) to generate a moons dataset. <br>
b) Split it into a training set and a test set using train_test_split().<br>
c) Use grid search with cross-validation (with the help of the GridSearchCV class) to find good hyperparameter values for a DecisionTreeClassifier.
Hint: try various values for max_leaf_nodes.<br>
d) Train it on the full training set using these hyperparameters, and measure your model’s performance on the test set. You should get roughly 85% to 87% accuracy. 

**2. Grow a forest:** <br>
a) Continuing the previous exercise, generate 1,000 subsets of the training set, each containing 100 instances selected randomly. Hint: you can use Scikit-Learn’s ShuffleSplit class for this. <br>
b) Train one Decision Tree on each subset, using the best hyperparameter values found above. Evaluate these 1,000 Decision Trees on the test set. Since they were trained on smaller sets, these Decision Trees will likely perform worse than the first Decision Tree, achieving only about 80% accuracy. <br>
c) Now comes the magic. For each test set instance, generate the predictions of the 1,000 Decision Trees, and keep only the most frequent prediction (you can use SciPy’s mode() function for this). This gives you majority-vote predictions over the test set. <br>
d) Evaluate these predictions on the test set: you should obtain a slightly higher accuracy than your first model (about 0.5 to 1.5% higher). Congratulations, you have trained a Random Forest classifier!

**3. Load the MNIST data and split it into a training set, a validation set, and a test set (e.g., use the first 40,000 instances for training, the next 10,000 for validation, and the last 10,000 for testing):** <br>
a) Then train various classifiers, such as a Random Forest classifier, an Extra-Trees classifier, and an SVM. <br>
b) Next, try to combine them into an ensemble that outperforms them all on the validation set, using a soft or hard voting classifier. Once you have found one, try it on the test set. How much better does it perform compared to the individual classifiers?

**4. Run the individual classifiers from the previous exercise to make predictions on the validation set:** <br>
a) Create a new training set with the resulting predictions: each training instance is a vector containing the set of predictions from all your classifiers for an image, and the target is the image’s class. Congratulations, you have just trained a blender, and together with the classifiers they form a stacking ensemble! <br>
b) Now let’s evaluate the ensemble on the test set. For each image in the test set, make predictions with all your classifiers, then feed the predictions to the blender to get the ensemble’s predictions. How does it compare to the voting classifier you trained earlier?
