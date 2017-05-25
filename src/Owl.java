 import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

public class Owl {

	public static Instances trainSet;
	public static Instances testSet;
	public static Classifier cls; 
	public static Instance inst;
	
	public static void main(String[] args) throws Exception {
		
		readInstances();
		System.out.println("Dataset loaded");
		//loadModel();
		System.out.println("Trained model loaded\n");
		System.out.println("Predictions: ");
		predict();
	
	}
	
	
	public static void readInstances() throws IOException{
		
		BufferedReader reader = new BufferedReader(new FileReader("dataset/train_edge.arff"));			
		trainSet = new Instances(reader);
		reader.close();
		
		reader = new BufferedReader(new FileReader("dataset/ugletest_filtered.arff"));	
		testSet = new Instances(reader);			
		reader.close();
	
		//setting class attribute
		trainSet.setClassIndex(trainSet.numAttributes() - 1);
		testSet.setClassIndex(testSet.numAttributes() - 1);

	}
	
	
	public static void loadModel() throws FileNotFoundException, IOException, ClassNotFoundException{

		File fil = new File("trainedModel/model_naiveBayes_edge.model");
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(fil.getAbsolutePath()));	
		cls = (Classifier) ois.readObject();
		ois.close();

	}
	
	public static void saveModel() throws FileNotFoundException, IOException{
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("/some/where/j48.model"));
		oos.writeObject(cls);				
		oos.flush();
		oos.close();
	}
	
	public static void predict() throws Exception{
		
		Remove rm = new Remove();
		rm.setAttributeIndices("1");  

		SMO vector = new SMO();
		MultilayerPerceptron neuralNets = new MultilayerPerceptron();
		J48 decisionTree = new J48();
		RandomForest forest = new RandomForest();

		
		//cls = forest;
		//cls = vector;
		//cls = neuralNets;
		//cls = decisionTree;
				
		 FilteredClassifier fc = new FilteredClassifier();
		 fc.setFilter(rm);
	     fc.setClassifier(cls);			 			
		 fc.buildClassifier(trainSet);
	   
		 Evaluation eval = new Evaluation(trainSet);
		 eval.evaluateModel(fc, testSet);
		
		 
		 for (int i = 0; i < trainSet.numInstances(); i++) {
		   double pred = fc.classifyInstance(trainSet.instance(i));		
		   System.out.print("actual: " + trainSet.classAttribute().value((int) trainSet.instance(i).classValue()));
		   System.out.println(", predicted: " + trainSet.classAttribute().value((int) pred));
		 }

		 System.out.println(eval.toSummaryString("\nResults Testdata\n======\n", false));
		 
		//saveModel();
	}

}
