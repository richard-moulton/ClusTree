/*
 *    ClusTree.java
 *    Copyright (C) 2010 RWTH Aachen University, Germany
 *    @author Sanchez Villaamil (moa@cs.rwth-aachen.de)
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *    
 *    
 */

package moa.clusterers.clustree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import moa.clusterers.clustree.util.*;
import moa.cluster.CFCluster;
import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.cluster.SphereCluster;
import moa.clusterers.AbstractClusterer;
import moa.core.FixedLengthList;
import moa.core.Measurement;
import moa.evaluation.MeasureCollection;
import moa.evaluation.SilhouetteCoefficient;
import moa.gui.visualization.DataPoint;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.github.javacliparser.FlagOption;
import com.yahoo.labs.samoa.instances.Instance;

/**
 * Citation: ClusTree: Philipp Kranen, Ira Assent, Corinna Baldauf, Thomas Seidl:
 * The ClusTree: indexing micro-clusters for anytime stream mining.
 * Knowl. Inf. Syst. 29(2): 249-272 (2011) 
*/
public class ClusTree extends AbstractClusterer{
	private static final long serialVersionUID = 1L;
	
	public IntOption horizonOption = new IntOption("horizon",
			'h', "Range of the window.", 1000);

    public IntOption maxHeightOption = new IntOption(
			"maxHeight", 'H',
			"The maximal height of the tree", getDefaultHeight());

	public FlagOption breadthFirstStrategyOption = new FlagOption(
			"breadthFirstStrategy", 'B',
			"Use breadth first strategy");
	
	public MultiChoiceOption offlineClusteringOption = new MultiChoiceOption("offlineClustering", 'o',
			"Choose the offline clustering algorithm to be used.", new String[]{"Silhouette k-means", "Ground truth k-means"},
			new String[]{"Uses the silhouette coefficient to determine the value of k.", "Uses the centres of the ground truth clusters."},0);
    
    protected int getDefaultHeight() {
    	return 8;
    }
    
    private static int INSERTIONS_BETWEEN_CLEANUPS = 10000;
    /**
     * The root node of the tree.
     */
    protected Node root;
    // Information about the data represented in this tree.
    /**
     * Dimensionality of the data points managed by this tree.
     */
    private int numberDimensions;
    /**
     * Parameter for the weighting function use to weight the entries.
     */
    protected double negLambda;
    /**
     * The current height of the tree. Should always be smaller than maxHeight.
     */
    private int height;
    /**
     * The maximal height of the tree.
     */
    protected int maxHeight;
    /**
     * This variable is used to keep the inverse height that is stored in every
     * node correct.
     */
    private int numRootSplits;
    /**
     * The threshold for the weighting of an Entry. An Entry is irrelevant, if
     * it is in a leaf and the weightedN of the data Cluster is smaller than
     * this threshold.
     * @see Entry#data
     */
    private double weightThreshold = 0.05;
    /**
     * Number of points inserted into the tree.
     */
    private int numberInsertions;
    private long timestamp;

    /**
     * Parameter to determine which insertion strategy to use
     */
    protected boolean breadthFirstStrat = false;
    
    //TODO: cleanup
    private Entry alsoUpdate;
    
    /**
     * Number of clusters to try to form when using the Silhouette k-means option.
     * Set to -1 if not initialized.
     */
    private int kOption;
    
    /**
     * A list of fixed length holding the last 100 points seen by the
     */
    private FixedLengthList<Instance> lastPoints;
    
    /**
     * Set to be equal to MOA's SilhouetteCoefficient.java class.
     * 
     * @see moa.evaluation.SilhouetteCoefficient
     */
    private double pointInclusionProbThreshold = 0.8;
    
    @Override
    public void resetLearningImpl() {
        breadthFirstStrat = breadthFirstStrategyOption.isSet();
        negLambda = (1.0 / (double) horizonOption.getValue())
                * (Math.log(weightThreshold) / Math.log(2));
        maxHeight = maxHeightOption.getValue();
        numberDimensions = -1;
        root = null;
        timestamp = 0;
        height = 0;
        numRootSplits = 0;
        numberInsertions = 0;
        kOption = -1;
        lastPoints = new FixedLengthList<Instance>(100);
    }


    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    public boolean isRandomizable() {
        return false;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    public double[] getVotesForInstance(Instance inst) {
        return null;
    }

    @Override
    public boolean implementsMicroClusterer() {
        return true;
    }



    @Override
    public void trainOnInstanceImpl(Instance instance) {
        timestamp++;
        //System.out.print("Instance "+timestamp+" added. numAttributes "+instance.numAttributes());
		//System.out.print(", classIndex "+instance.classIndex());
		//System.out.println(", classValue "+instance.classValue());
        this.lastPoints.add(instance);
        //TODO check if instance contains label
        if(root == null){
            numberDimensions = instance.numAttributes();
            root = new Node(numberDimensions, 0);
        }
        else{
            if(numberDimensions!=instance.numAttributes())
                System.out.println("Wrong dimensionality, expected:"+numberDimensions+ "found:"+instance.numAttributes());
        }

        ClusKernel newPointAsKernel = new ClusKernel(instance.toDoubleArray(), numberDimensions);
        insert(newPointAsKernel, new SimpleBudget(1000),timestamp);
    }


    /**
     * Insert a new point in the <code>Tree</code>. The point should be 
     * represented as a cluster with a single data point(i.e. N = 1). A
     * <code>Budget</code> class is also given, which is informed of the number
     * of operation the tree does, and informs the tree when it does not have
     * time left and should stop the insertion.
     * @param newPoint The point to be inserted.
     * @param budget The budget and statistics recollector for the insertion.
     * @param timestamp The moment at which this point is inserted.
     * @see Kernel
     * @see Budget
     */
    public void insert(ClusKernel newPoint, Budget budget, long timestamp) {
        if (breadthFirstStrat){
          	insertBreadthFirst(newPoint, budget, timestamp);
       }
        else{
	    	Entry rootEntry = new Entry(this.numberDimensions,
	                root, timestamp, null, null);
	        ClusKernel carriedBuffer = new ClusKernel(this.numberDimensions);
	        Entry toInsertHere = insert(newPoint, carriedBuffer, root, rootEntry,
	                budget, timestamp);
	
	        if (toInsertHere != null) {
	            this.numRootSplits++;
	            this.height += this.height < this.maxHeight ? 1 : 0;
	            
	            Node newRoot = new Node(this.numberDimensions,
	                    toInsertHere.getChild().getRawLevel() + 1);
	            newRoot.addEntry(rootEntry, timestamp);
	            newRoot.addEntry(toInsertHere, timestamp);
	            rootEntry.setNode(newRoot);
	            toInsertHere.setNode(newRoot);
	            this.root = newRoot;
	        }
        }

        this.numberInsertions++;
        if (this.numberInsertions % INSERTIONS_BETWEEN_CLEANUPS == 0) {
            cleanUp(this.root, 0);
        }
    }

    /**
     * insert newPoint into the tree using the BreadthFirst strategy, i.e.: insert into
     * the closest entry in a leaf node.
     * @param newPoint
     * @param budget
     * @param timestamp
     * @return
     */
	private Entry insertBreadthFirst(ClusKernel newPoint, Budget budget, long timestamp) {
    	//check all leaf nodes and get the one with the closest entry to newPoint
		Node bestFit = findBestLeafNode(newPoint);
    	bestFit.makeOlder(timestamp, negLambda);
    	Entry parent = bestFit.getEntries()[0].getParentEntry(); 	
    	// Search for an Entry with a weight under the threshold.
   	    Entry irrelevantEntry = bestFit.getIrrelevantEntry(this.weightThreshold);
        int numFreeEntries = bestFit.numFreeEntries();
        Entry newEntry = new Entry(newPoint.getCenter().length,
    			newPoint, timestamp, parent, bestFit);
        //if there is space, add it to the node ( doesn't ever occur, since nodes are created with 3 entries) 
        if (numFreeEntries>0){
        	bestFit.addEntry(newEntry, timestamp);
        }
        //if outdated cluster in this best fitting node, replace it
         else if (irrelevantEntry != null) {
	        irrelevantEntry.overwriteOldEntry(newEntry);
        }
        //if there is space/outdated cluster on path to top, split. Else merge without split
        else {
        	if (existsOutdatedEntryOnPath(bestFit)||!this.hasMaximalSize()){
	            // We have to split.
	        	insertHereWithSplit(newEntry, bestFit, timestamp);
	        }
        	else {
	            mergeEntryWithoutSplit(bestFit, newEntry,
	                    timestamp);
	        }
        }
        //update all nodes on path to top.
        if (bestFit.getEntries()[0].getParentEntry()!=null)
        	updateToTop(bestFit.getEntries()[0].getParentEntry().getNode());
        return null;
    }
 /**
  * This method checks if there is an outdated (or empty) entry on the path from node to root.
  * It updates the weights of nodes on path and then checks if it is outdated.
  * @param node
  * @return true if an outdated/empty entry exists on the path
  */
	private boolean existsOutdatedEntryOnPath(Node node) {
    	if (node == root){
    		node.makeOlder(timestamp, negLambda);
    		return node.getIrrelevantEntry(this.weightThreshold)!=null;
    	}
    	do{
    		node = node.getEntries()[0].getParentEntry().getNode();
    		node.makeOlder(timestamp, negLambda);
    		for (Entry e : node.getEntries()){
    			e.recalculateData();
    		}
    		if (node.numFreeEntries()>0)
    			return true;
    		if (node.getIrrelevantEntry(this.weightThreshold)!=null)
    			return true;
    	}while(node.getEntries()[0].getParentEntry()!=null);
		return false;
	}
    
    /**
     * recalculates data for all entries, that lie on the path from the root to the 
     * Entry toUpdate.
     */
	private void updateToTop(Node toUpdate) {
    	while(toUpdate!=null){
    		for (Entry e: toUpdate.getEntries())
    			e.recalculateData();
    		if (toUpdate.getEntries()[0].getParentEntry()==null)
    			break;
    		toUpdate=toUpdate.getEntries()[0].getParentEntry().getNode();
    	}
	}

	/**
	 * Method called by insertBreadthFirst.
	 * @param toInsert
	 * @param insertNode
	 * @param timestamp
	 * @return
	 */
	private Entry insertHereWithSplit(Entry toInsert, Node insertNode,
			long timestamp) {
		//Handle root split
   	    if (insertNode.getEntries()[0].getParentEntry()==null){
   	    	root.makeOlder(timestamp, negLambda);
   	    	Entry irrelevantEntry = insertNode.getIrrelevantEntry(this.weightThreshold);
   	        int numFreeEntries = insertNode.numFreeEntries();
   	        if (irrelevantEntry != null) {
   		        irrelevantEntry.overwriteOldEntry(toInsert);
   	        }
   	        else if (numFreeEntries>0){
   	        	insertNode.addEntry(toInsert, timestamp);
   	        }
   	        else{
	            this.numRootSplits++;
	            this.height += this.height < this.maxHeight ? 1 : 0;
	            Entry oldRootEntry = new Entry(this.numberDimensions,
	                    root, timestamp, null, null);
	            Node newRoot = new Node(this.numberDimensions,
	                    this.height);
	            Entry newRootEntry = split(toInsert, root, oldRootEntry, timestamp);
	            newRoot.addEntry(oldRootEntry, timestamp);
	            newRoot.addEntry(newRootEntry, timestamp);
	            this.root = newRoot;
	            for (Entry c : oldRootEntry.getChild().getEntries())
	            	c.setParentEntry(root.getEntries()[0]);
	            for (Entry c : newRootEntry.getChild().getEntries())
	            	c.setParentEntry(root.getEntries()[1]);
   	        }
            return null;
   	    }
   	    insertNode.makeOlder(timestamp, negLambda);
    	Entry irrelevantEntry = insertNode.getIrrelevantEntry(this.weightThreshold);
        int numFreeEntries = insertNode.numFreeEntries();
        if (irrelevantEntry != null) {
	        irrelevantEntry.overwriteOldEntry(toInsert);
        }
        else if (numFreeEntries>0){
        	insertNode.addEntry(toInsert, timestamp);
        }
        else {
	            // We have to split.
	        	Entry parentEntry = insertNode.getEntries()[0].getParentEntry();
	        	Entry residualEntry = split(toInsert, insertNode, parentEntry, timestamp);
	        	if (alsoUpdate!=null){
	        		alsoUpdate = residualEntry;
	        	}
	        	Node nodeForResidualEntry = insertNode.getEntries()[0].getParentEntry().getNode();
	        	//recursive call
	        	return insertHereWithSplit(residualEntry, nodeForResidualEntry, timestamp);
	    }
        
        //no Split
        return null;
	}


	// XXX: Document the insertion when the final implementation is done.
	private Entry insertHere(Entry newEntry, Node currentNode,
	        Entry parentEntry, ClusKernel carriedBuffer, Budget budget,
	        long timestamp) {
	
	    int numFreeEntries = currentNode.numFreeEntries();
	
	    // Insert the buffer that we carry.
	    if (!carriedBuffer.isEmpty()) {
	        Entry bufferEntry = new Entry(this.numberDimensions,
	                carriedBuffer, timestamp, parentEntry, currentNode);
	
	        if (numFreeEntries <= 1) {
	            // Distance from buffer to entries.
	            Entry nearestEntryToCarriedBuffer =
	                    currentNode.nearestEntry(newEntry);
	            double distanceNearestEntryToBuffer =
	                    nearestEntryToCarriedBuffer.calcDistance(newEntry);
	
	            // Distance between buffer and point to insert.
	            double distanceBufferNewEntry =
	                    newEntry.calcDistance(carriedBuffer);
	
	            // Best distance between Entrys in the Node.
	            BestMergeInNode bestMergeInNode =
	                    calculateBestMergeInNode(currentNode);
	
	            // See what the minimal distance is and do the correspoding
	            // action.
	            if (distanceNearestEntryToBuffer <= distanceBufferNewEntry
	                    && distanceNearestEntryToBuffer <= bestMergeInNode.distance) {
	                // Aggregate buffer entry to nearest entry in node.
	                nearestEntryToCarriedBuffer.aggregateEntry(bufferEntry,
	                        timestamp, this.negLambda);
	            } else if (distanceBufferNewEntry <= distanceNearestEntryToBuffer
	                    && distanceBufferNewEntry <= bestMergeInNode.distance) {
	                newEntry.mergeWith(bufferEntry);
	            } else {
	                currentNode.mergeEntries(bestMergeInNode.entryPos1,
	                        bestMergeInNode.entryPos2);
	                currentNode.addEntry(bufferEntry, timestamp);
	            }
	
	        } else {
	            assert (currentNode.isLeaf());
	            currentNode.addEntry(bufferEntry, timestamp);
	        }
	    }
	
	    // Normally the insertion of the carries buffer does not change the
	    // number of free entries, but in case of future changes we calculate
	    // the number again.
	    numFreeEntries = currentNode.numFreeEntries();
	
	    // Search for an Entry with a weight under the threshold.
	    Entry irrelevantEntry = currentNode.getIrrelevantEntry(this.weightThreshold);
	    if (currentNode.isLeaf() && irrelevantEntry != null) {
	        irrelevantEntry.overwriteOldEntry(newEntry);
	    } else if (numFreeEntries >= 1) {
	        currentNode.addEntry(newEntry, timestamp);
	    } else {
	        if (currentNode.isLeaf() && (this.hasMaximalSize()
	                || !budget.hasMoreTime())) {
	            mergeEntryWithoutSplit(currentNode, newEntry,
	                    timestamp);
	        } else {
	            // We have to split.
	            return split(newEntry, currentNode, parentEntry, timestamp);
	        }
	    }
	
	    return null;
	}

	/**
	 * This method calculates the distances between the new point and each Entry in a leaf node.
	 * It returns the node that contains the entry with the smallest distance
	 * to the new point.
	 * @param newPoint
	 * @return best fitting node
	 */
	private Node findBestLeafNode(ClusKernel newPoint) {
    	double minDist = Double.MAX_VALUE;
    	Node bestFit = null;
    	for (Node e: collectLeafNodes(root)){
    		if (newPoint.calcDistance(e.nearestEntry(newPoint).getData())<minDist){
    			bestFit = e;
    			minDist = newPoint.calcDistance(e.nearestEntry(newPoint).getData());
    		}
    	}
    	if (bestFit!=null)
    		return bestFit;
    	else
    		return root;
	}
    
    private ArrayList<Node> collectLeafNodes(Node curr){
    	ArrayList<Node> toReturn = new ArrayList<Node>();
    	if (curr==null)
    		return toReturn;
    	if	(curr.isLeaf()){
    		toReturn.add(curr);
    		return toReturn;
    	}
    	else{
    		for (Entry e : curr.getEntries())
    			toReturn.addAll(collectLeafNodes(e.getChild()));
    		return toReturn;
    	}
    }

	// TODO: Expand all function that work on entries to work with the Budget.
    private Entry insert(ClusKernel pointToInsert, ClusKernel carriedBuffer,
            Node currentNode, Entry parentEntry, Budget budget, long timestamp) {
        assert (currentNode != null);
        assert (currentNode.isLeaf()
                || currentNode.getEntries()[0].getChild() != null);

        currentNode.makeOlder(timestamp, this.negLambda);

        // This variable will be changed from to null to an actual reference
        // in the following if-else block if we have to insert something here,
        // either because this is a leaf, or because of split propagation.
        Entry toInsertHere = null;

        if (currentNode.isLeaf()) {
            // At the end of the function the entry will be inserted.
            toInsertHere = new Entry(this.numberDimensions,
                    pointToInsert, timestamp, parentEntry, currentNode);
        } else {

            Entry bestEntry = currentNode.nearestEntry(pointToInsert);
            bestEntry.aggregateCluster(pointToInsert, timestamp,
                    this.negLambda);

            boolean isCarriedBufferEmpty = carriedBuffer.isEmpty();

            Entry bestBufferEntry = null;
            if (!isCarriedBufferEmpty) {
                bestBufferEntry = currentNode.nearestEntry(carriedBuffer);
                bestBufferEntry.aggregateCluster(carriedBuffer, timestamp,
                        this.negLambda);
            }

            if (!budget.hasMoreTime()) {
                bestEntry.aggregateToBuffer(pointToInsert, timestamp,
                        this.negLambda);
                if (!isCarriedBufferEmpty) {
                    bestBufferEntry.aggregateToBuffer(carriedBuffer,
                            timestamp, this.negLambda);
                }
                return null;
            }

            // If the way of the buffer differs from the way of the point to
            // be inserted, leave the buffer here.
            if (!isCarriedBufferEmpty && (bestEntry != bestBufferEntry)) {
                bestBufferEntry.aggregateToBuffer(carriedBuffer, timestamp,
                        this.negLambda);
                carriedBuffer.clear();
            }
            // Take the buffer of the best entry for the point to be inserted
            // along.
            ClusKernel takeAlongBuffer = bestEntry.emptyBuffer(timestamp,
                    this.negLambda);
            carriedBuffer.add(takeAlongBuffer);

            // Recursive call.
            toInsertHere = insert(pointToInsert, carriedBuffer,
                    bestEntry.getChild(), bestEntry, budget, timestamp);
        }

        // If the above block has a new Entry for this place insert it.
        if (toInsertHere != null) {
            return this.insertHere(toInsertHere, currentNode, parentEntry,
                    carriedBuffer, budget, timestamp);
        }

        // If nothing else needs to be done in all the above levels
        // return null to signalize it.
        return null;
    }

    /**
     * Inserts an <code>Entry</code> into a <code>Node</code> without inducing
     * a split.
     * @param node The node at which the entry is to be inserted.
     * @param newEntry The entry to be inserted.
     * @param timestamp The moment at which this occurs.
     */
    private void mergeEntryWithoutSplit(Node node,
            Entry newEntry, long timestamp) {

        Entry nearestEntryToCarriedBuffer =
                node.nearestEntry(newEntry);
        double distanceNearestEntryToBuffer =
                nearestEntryToCarriedBuffer.calcDistance(newEntry);

        BestMergeInNode bestMergeInNode =
                calculateBestMergeInNode(node);

        if (distanceNearestEntryToBuffer < bestMergeInNode.distance) {
            nearestEntryToCarriedBuffer.aggregateEntry(newEntry, timestamp,
                    this.negLambda);
        } else {
            node.mergeEntries(bestMergeInNode.entryPos1,
                    bestMergeInNode.entryPos2);
            node.addEntry(newEntry, timestamp);
        }
    }

    /**
     * Calculates the best merge possible between two nodes in a node. This
     * means that the pair with the smallest distance is found.
     * @param node The node in which these two entries have to be found.
     * @return An object which encodes the two position of the entries with the
     * smallest distance in the node and the distance between them.
     * @see BestMergeInNode
     * @see Entry#calcDistance(tree.Entry) 
     */
    private BestMergeInNode calculateBestMergeInNode(Node node) {
        assert (node.numFreeEntries() == 0);

        Entry[] entries = node.getEntries();

        int toMerge1 = -1;
        int toMerge2 = -1;
        double distanceBetweenMergeEntries = Double.NaN;

        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < entries.length; i++) {
            Entry e1 = entries[i];
            for (int j = i + 1; j < entries.length; j++) {
                Entry e2 = entries[j];
                double distance = e1.calcDistance(e2);
                if (distance < minDistance) {
                    toMerge1 = i;
                    toMerge2 = j;
                    distanceBetweenMergeEntries = distance;
                }
            }
        }

        assert (toMerge1 != -1 && toMerge2 != -1);
        if (Double.isNaN(distanceBetweenMergeEntries)) {
            throw new RuntimeException("The minimal distance between two "
                    + "Entrys in a Node was Double.MAX_VAUE. That can hardly "
                    + "be right.");
        }

        return new BestMergeInNode(toMerge1, toMerge2,
                distanceBetweenMergeEntries);
    }

    private boolean hasMaximalSize() {
        // TODO: Improve hasMaximalSize(). For now it just works somehow for testing.
        return this.height == this.maxHeight;
    }

    /**
     * Performs a (2,2) split on the given node with the given entry. This
     * implementation only works if the nodes have three entries each. The split
     * will generate two new nodes. One of them will be put where the old node
     * was, and for the other a new <code>Entry</code> will be generated and
     * returned.
     * @param newEntry The entry to be added to the node.
     * @param node The node that is going to be splitted.
     * @param parentEntry The entry in the tree that points at the node that
     * is going to be splitted.
     * @param timestamp The moment at which this split occurs.
     * @return An entry which points at the second node created in the split.
     * This entry has to be introduced later in the tree.
     */
    private Entry split(Entry newEntry, Node node, Entry parentEntry,
            long timestamp) {
        // The implemented split function only works in trees where node
        // have three entries.
        // Splitting only makes sense on full nodes.
        assert (node.numFreeEntries() == 0);
        assert (parentEntry.getChild() == node);

        // All the entries we have to separate in two nodes.
        Entry[] allEntries = new Entry[4];
        Entry[] nodeEntries = node.getEntries();
        for (int i = 0; i < nodeEntries.length; i++) {
            allEntries[i] = new Entry(nodeEntries[i]);
        }
        allEntries[3] = newEntry;

        // Clear the given node, since we are going to refill it later.
        node = new Node(this.numberDimensions, node.getRawLevel());

        // Calculate the distance of all the possible pairings, since we want
        // to do a (2,2) split.
        double select01 = allEntries[0].calcDistance(allEntries[1])
                + allEntries[2].calcDistance(allEntries[3]);

        double select02 = allEntries[0].calcDistance(allEntries[2])
                + allEntries[1].calcDistance(allEntries[3]);

        double select03 = allEntries[0].calcDistance(allEntries[3])
                + allEntries[1].calcDistance(allEntries[2]);

        // See which of the pairings is minimal and distribute the entries
        // accordingly.
        Node residualNode = new Node(this.numberDimensions,
                node.getRawLevel());
        if (select01 < select02) {
            if (select01 < select03) {//select01 smallest
                node.addEntry(allEntries[0], timestamp);
                node.addEntry(allEntries[1], timestamp);
                residualNode.addEntry(allEntries[2], timestamp);
                residualNode.addEntry(allEntries[3], timestamp);
            } else {//select03 smallest
                node.addEntry(allEntries[0], timestamp);
                node.addEntry(allEntries[3], timestamp);
                residualNode.addEntry(allEntries[1], timestamp);
                residualNode.addEntry(allEntries[2], timestamp);
            }
        } else {
            if (select02 < select03) {//select02 smallest
                node.addEntry(allEntries[0], timestamp);
                node.addEntry(allEntries[2], timestamp);
                residualNode.addEntry(allEntries[1], timestamp);
                residualNode.addEntry(allEntries[3], timestamp);
            } else {//select03 smallest
                node.addEntry(allEntries[0], timestamp);
                node.addEntry(allEntries[3], timestamp);
                residualNode.addEntry(allEntries[1], timestamp);
                residualNode.addEntry(allEntries[2], timestamp);
            }
        }

        // Set the other node into the tree.
        parentEntry.setChild(node);
        parentEntry.recalculateData();
        int count = 0;
        for (Entry e : node.getEntries()){
        	e.setParentEntry(parentEntry);
        	if (e.getData().getN() != 0)
        		count++;
        }
        //System.out.println(count);
        // Generate a new entry for the residual node.
        Entry residualEntry = new Entry(this.numberDimensions,
                residualNode, timestamp, parentEntry, node);
        count=0;
        for (Entry e: residualNode.getEntries()){
        	e.setParentEntry(residualEntry);
        	if (e.getData().getN() != 0)
        		count++;
        }
        //System.out.println(count);
        return residualEntry;
    }

    /**
     * Return the number of time the tree has grown in size. If the tree grows
     * and is then cutted from a certain depth, it also counts.
     * @return The number of times the root node was splitted.
     */
    public int getNumRootSplits() {
        return numRootSplits;
    }

    /**
     * Return the current height of the tree. This should never be greater than
     * <code>maxHeight</code>.
     * @return The height of the tree.
     * @see #maxHeight
     */
    public int getHeight() {
        assert (height <= maxHeight);
        return height;
    }

    private void cleanUp(Node currentNode, int level) {
        if (currentNode == null) {
            return;
        }

        Entry[] entries = currentNode.getEntries();
        if (level == this.maxHeight) {
            for (int i = 0; i < entries.length; i++) {
                Entry e = entries[i];
                e.setChild(null);
            }
        } else {
            for (int i = 0; i < entries.length; i++) {
                Entry e = entries[i];
                cleanUp(e.getChild(), level + 1);
            }
        }
    }

    /**
     * @param currentTime The current time
     * @return The kernels at the leaf level as a clustering
     */
    //TODO: Microcluster unter dem Threshhold nich zur�ckgeben (WIe bei outdated entries)
    @Override
    public Clustering getMicroClusteringResult() {
        return getClustering(timestamp, -1);
    }

    @Override
    public Clustering getClusteringResult()
    {
    	if (this.offlineClusteringOption.getChosenLabel().equals("Silhouette k-means"))
    	{
    		if (this.root == null)
    		{
    			return new Clustering(new Cluster[0]);
    		}
    		
    		Clustering microClusters = getMicroClusteringResult();
    		
    		updateKOption(microClusters);
    		   		
    		return kMeans_rand(this.kOption, microClusters);
    	}
    	
    	// Else offlineClusteringOption is "Ground truth k-means"
    	return null;
    }


    /**
     * @param currentTime The current time
     * @return The kernels at the given level as a clustering.
     */
    public Clustering getClustering(long currentTime, int targetLevel) {
        if (root == null) {
            return null;
        }

        Clustering clusters = new Clustering();
        LinkedList<Node> queue = new LinkedList<Node>();
        queue.add(root);

        while (!queue.isEmpty()) {
            Node current = queue.remove();
           // if (current == null)
           // 	continue;
            int currentLevel = current.getLevel(this);
            boolean isLeaf = (current.isLeaf() && currentLevel <= maxHeight)
                    || currentLevel == maxHeight;

            if (currentLevel == targetLevel
                    || (targetLevel == - 1 && isLeaf)) {
                assert (currentLevel <= maxHeight);

                Entry[] entries = current.getEntries();
                for (int i = 0; i < entries.length; i++) {
                    Entry entry = entries[i];
                    if (entry == null || entry.isEmpty()) {
                        continue;
                    }
                    // XXX
                    entry.makeOlder(currentTime, this.negLambda);
                    if (entry.isIrrelevant(this.weightThreshold))
                    	continue;

                    ClusKernel gaussKernel = new ClusKernel(entry.getData());

//                  long diff = currentTime - entry.getTimestamp();
//                    if (diff > 0) {
//                        gaussKernel.makeOlder(diff, negLambda);
//                    }

                    clusters.add(gaussKernel);
                }
            } else if (!current.isLeaf()) {
                Entry[] entries = current.getEntries();
                for (int i = 0; i < entries.length; i++) {
                    Entry entry = entries[i];

                    if (entry.isEmpty()) {
                        continue;
                    }

                    if (entry.isIrrelevant(weightThreshold)) {
                        continue;
                    }

                    queue.add(entry.getChild());
                }
            }
        }

        return clusters;
    }



    /**************************************************************************
     * LOCAL CLASSES
     **************************************************************************/
    /**
     * A class to code the return value of searching the smallest merge in a
     * node.
     */
    class BestMergeInNode {

        /**
         * The position of the first entry in the array of the node.
         */
        public int entryPos1;
        /**
         * The position of the second entry in the array of the node.
         */
        public int entryPos2;
        /**
         * The distance between the two entries.
         */
        public double distance;

        /**
         * The constructor of this return value. It will automatically make
         * sure that the first position is the smaller one of the two.
         * @param pos1 One of the position.
         * @param pos2 One of the position.
         * @param distance The distance between the entries at these positions.
         */
        public BestMergeInNode(int pos1, int pos2,
                double distance) {
            assert (pos1 != pos2);

            this.distance = distance;

            if (pos1 < pos2) {
                this.entryPos1 = pos1;
                this.entryPos2 = pos2;
            } else {
                this.entryPos1 = pos2;
                this.entryPos2 = pos1;
            }
        }
    }

    /**
	 * k-means of (micro)clusters, with randomized initialization. 
	 * 
	 * @param k the number of clusters to find
	 * @param clustering the microclusters
	 * @return (macro)clustering - CFClusters
	 */
	public static Clustering kMeans_rand(int k, Clustering clustering)
	{
		
		ArrayList<CFCluster> microclusters = new ArrayList<CFCluster>();
        for (int i = 0; i < clustering.size(); i++) {
            if (clustering.get(i) instanceof CFCluster) {
                microclusters.add((CFCluster)clustering.get(i));
            } else {
                System.out.println("Unsupported Cluster Type:" + clustering.get(i).getClass() + ". Cluster needs to extend moa.cluster.CFCluster");
            }
        }
        
        int n = microclusters.size();
		assert (k <= n);
		
		/* k-means */
		Random random = new Random(0);
		Cluster[] centers = new Cluster[k];
		
		for (int i = 0; i < k; i++) {
			int rid = random.nextInt(n);
			centers[i] = new SphereCluster(microclusters.get(rid).getCenter(), 0);
		}
		
		return cleanUpKMeans(kMeans(k, centers, microclusters), microclusters);
	}
	
	/**
	 * (The Actual Algorithm) k-means of (micro)clusters, with specified initialization points.
	 * 
	 * @param k the number of clusters to find
	 * @param centers the initial centers
	 * @param data the microclusters
	 * @return (macro)clustering - SphereClusters
	 */
	protected static Clustering kMeans(int k, Cluster[] centers, List<? extends Cluster> data) {
		assert (centers.length == k);
		assert (k > 0);

		int dimensions = centers[0].getCenter().length;

		ArrayList<ArrayList<Cluster>> clustering = new ArrayList<ArrayList<Cluster>>();
		for (int i = 0; i < k; i++) {
			clustering.add(new ArrayList<Cluster>());
		}

		while (true) {
			// Assign points to clusters
			for (Cluster point : data) {
				double minDistance = distance(point.getCenter(), centers[0].getCenter());
				int closestCluster = 0;
				for (int i = 1; i < k; i++) {
					double distance = distance(point.getCenter(), centers[i].getCenter());
					if (distance < minDistance) {
						closestCluster = i;
						minDistance = distance;
					}
				}

				clustering.get(closestCluster).add(point);
			}

			// Calculate new centers and clear clustering lists
			SphereCluster[] newCenters = new SphereCluster[centers.length];
			for (int i = 0; i < k; i++) {
				newCenters[i] = calculateCenter(clustering.get(i), dimensions);
				clustering.get(i).clear();
			}
			
			// Convergence check
			boolean converged = true;
			for (int i = 0; i < k; i++) {
				if (!Arrays.equals(centers[i].getCenter(), newCenters[i].getCenter())) {
					converged = false;
					break;
				}
			}
			
			if (converged) {
				break;
			} else {
				centers = newCenters;
			}
		}

		return new Clustering(centers);
	}
	
	/**
	 * Rearrange the k-means result into a set of CFClusters, cleaning up the redundancies.
	 * 
	 * @param kMeansResult the macroclustering produced by k-means
	 * @param microclusters the microclusters produced by ClusTree
	 * @return the cleaned macroclustering
	 */
	protected static Clustering cleanUpKMeans(Clustering kMeansResult, ArrayList<CFCluster> microclusters) {
		/* Convert k-means result to CFClusters */
		int k = kMeansResult.size();
		CFCluster[] converted = new CFCluster[k];

		for (CFCluster mc : microclusters) {
		    // Find closest kMeans cluster
		    double minDistance = Double.MAX_VALUE;
		    int closestCluster = 0;
		    for (int i = 0; i < k; i++) {
		    	double distance = distance(kMeansResult.get(i).getCenter(), mc.getCenter());
				if (distance < minDistance) {
				    closestCluster = i;
				    minDistance = distance;
				}
		    }

		    // Add to cluster
		    if ( converted[closestCluster] == null ) {
		    	converted[closestCluster] = (CFCluster)mc.copy();
		    } else {
		    	converted[closestCluster].add(mc);
		    }
		}

		// Clean up
		int count = 0;
		for (int i = 0; i < converted.length; i++) {
		    if (converted[i] != null)
			count++;
		}

		CFCluster[] cleaned = new CFCluster[count];
		count = 0;
		for (int i = 0; i < converted.length; i++) {
		    if (converted[i] != null)
			cleaned[count++] = converted[i];
		}

		return new Clustering(cleaned);
	}

	

	/**
	 * k-means helper: Calculate a wrapping cluster of assigned points[microclusters].
	 * 
	 * @param assigned
	 * @param dimensions
	 * @return SphereCluster (with center and radius)
	 */
	private static SphereCluster calculateCenter(ArrayList<Cluster> assigned, int dimensions) {
		double[] result = new double[dimensions];
		for (int i = 0; i < result.length; i++) {
			result[i] = 0.0;
		}

		if (assigned.size() == 0) {
			return new SphereCluster(result, 0.0);
		}

		for (Cluster point : assigned) {
			double[] center = point.getCenter();
			for (int i = 0; i < result.length; i++) {
				result[i] += center[i];
			}
		}

		// Normalize
		for (int i = 0; i < result.length; i++) {
			result[i] /= assigned.size();
		}

		// Calculate radius: biggest wrapping distance from center
		double radius = 0.0;
		for (Cluster point : assigned) {
			double dist = distance(result, point.getCenter());
			if (dist > radius) {
				radius = dist;
			}
		}
		SphereCluster sc = new SphereCluster(result, radius);
		sc.setWeight(assigned.size());
		return sc;
	}
	
	/**
	 * Euclidean distance between two vectors.
	 * 
	 * @param pointA
	 * @param pointB
	 * @return dist
	 */
	private static double distance(double[] pointA, double [] pointB) {
		double distance = 0.0;
		for (int i = 0; i < pointA.length; i++) {
			double d = pointA[i] - pointB[i];
			distance += d * d;
		}
		return Math.sqrt(distance);
	}
	
	/**
	 * Updates the global variable kOption based on the current value of kOption and the
	 * clusterings produced from the argument microclusters.
	 * 
	 * @param clustering the microclusters produced by ClusTree.
	 */
	private void updateKOption(Clustering clustering)
	{
		//System.out.print("Update k option");
		Clustering clusResult;
		double bestSil = -Double.MAX_VALUE;
		double currSil;
		int bestK = -1;
		ArrayList<DataPoint> dataPoints = new ArrayList<DataPoint>();
		
		for(int i = 0 ; i < lastPoints.size() ; i++)
		{
			dataPoints.add(new DataPoint(lastPoints.get(i), i));
			//System.out.print("Data point "+i+" added. numAttributes "+dataPoints.get(i).numAttributes());
			//System.out.print(", classIndex "+dataPoints.get(i).classIndex());
			//System.out.println(", classValue "+dataPoints.get(i).classValue());
		}
		
		
		if(this.kOption == -1)
		{
			//System.out.println(" -1//");

			int n = (int)Math.sqrt(((double)clustering.size())/2.0);
			
			for(int i = Math.max(2, n-10) ; i < (n + 10) ; i++)
			{
				clusResult = kMeans_rand(i,clustering);
				currSil = getSilhouetteCoefficient(clusResult, dataPoints);
				if(currSil > bestSil)
				{
					bestSil = currSil;
					bestK = i;
				}
				//System.out.print(" k="+i+" sil="+currSil+" ("+bestK+"/"+bestSil+"). ");
			}
		}
		else
		{
			//System.out.print(" +");
			for(int i = Math.max(2, this.kOption - 4) ; i < (this.kOption + 4) ; i++)
			{
				clusResult = kMeans_rand(i,clustering);
				currSil = getSilhouetteCoefficient(clusResult, dataPoints);
				if(currSil > bestSil)
				{
					bestSil = currSil;
					bestK = i;
				}
				//System.out.print(" k="+i+" sil="+currSil+" ("+bestK+"/"+bestSil+"). ");
			}
		}
		
		this.kOption = bestK;
	}
	
	private double getSilhouetteCoefficient(Clustering clustering, ArrayList<DataPoint> points)
	{
    	int numFCluster = clustering.size();

        double [][] pointInclusionProbFC = new double[points.size()][numFCluster];
        for (int p = 0; p < points.size(); p++) {
            DataPoint point = points.get(p);
            for (int fc = 0; fc < numFCluster; fc++) {
                Cluster cl = clustering.get(fc);
                pointInclusionProbFC[p][fc] = cl.getInclusionProbability(point);
            }
        }

        double silhCoeff = 0.0;
        int totalCount = 0;
        for (int p = 0; p < points.size(); p++) {
            DataPoint point = points.get(p);
            ArrayList<Integer> ownClusters = new ArrayList<Integer>();
            for (int fc = 0; fc < numFCluster; fc++) {
                if(pointInclusionProbFC[p][fc] > pointInclusionProbThreshold){
                    ownClusters.add(fc);
                }
            }

            if(ownClusters.size() > 0){
                double[] distanceByClusters = new double[numFCluster];
                int[] countsByClusters = new int[numFCluster];
                    //calculate averageDistance of p to all cluster
                for (int p1 = 0; p1 < points.size(); p1++) {
                    DataPoint point1 = points.get(p1);
                    if(p1!= p){
                        for (int fc = 0; fc < numFCluster; fc++) {
                            if(pointInclusionProbFC[p1][fc] > pointInclusionProbThreshold){
                                double distance = distance(point.toDoubleArray(), point1.toDoubleArray());
                                distanceByClusters[fc]+=distance;
                                countsByClusters[fc]++;
                            }
                        }
                    }
                }

                //find closest OWN cluster as clusters might overlap
                double minAvgDistanceOwn = Double.MAX_VALUE;
                int minOwnIndex = -1;
                for (int fc : ownClusters) {
                        double normDist = distanceByClusters[fc]/(double)countsByClusters[fc];
                        if(normDist < minAvgDistanceOwn){// && pointInclusionProbFC[p][fc] > pointInclusionProbThreshold){
                            minAvgDistanceOwn = normDist;
                            minOwnIndex = fc;
                        }
                }


                //find closest other (or other own) cluster
                double minAvgDistanceOther = Double.MAX_VALUE;
                for (int fc = 0; fc < numFCluster; fc++) {
                    if(fc != minOwnIndex){
                        double normDist = distanceByClusters[fc]/(double)countsByClusters[fc];
                        if(normDist < minAvgDistanceOther){
                            minAvgDistanceOther = normDist;
                        }
                    }
                }

                double silhP = (minAvgDistanceOther-minAvgDistanceOwn)/Math.max(minAvgDistanceOther, minAvgDistanceOwn);
                point.setMeasureValue("SC - own", minAvgDistanceOwn);
                point.setMeasureValue("SC - other", minAvgDistanceOther);
                point.setMeasureValue("SC", silhP);

                silhCoeff+=silhP;
                totalCount++;
                //System.out.println(point.getTimestamp()+" Silh "+silhP+" / "+minAvgDistanceOwn+" "+minAvgDistanceOther+" (C"+minOwnIndex+")");
            }
        }
        if(totalCount>0)
            silhCoeff/=(double)totalCount;
        //normalize from -1, 1 to 0,1
        silhCoeff = (silhCoeff+1)/2.0;
        
        return silhCoeff;
    }
}
