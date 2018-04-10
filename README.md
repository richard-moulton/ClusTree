# ClusTree
Modifications to the ClusTree clustering algorithm implemented in MOA.

The ClusTree algorithm is described by Kranen et al. in their paper "The ClusTree: indexing micro-clusters for anytime stream mining" [2]. Please cite that paper if you use this code.

The modifications to this code implement the **getClusteringResult()** method. Previously MOA would apply k-means guided by the ground truth to the microclusters produced by ClusTree. Now, as described by Kranen et al., the microclusters can be properly clustered using k-means. The choice of *k* is determined first by evaluating the Silhouette coefficient of clusterings produced by a wide range of potential *k* values. Subsqeuent *k* values are selected by similarly evaluating clustering produced by a restricted range of values centered on the current *k* value.

This modification also contributes a new data structure: the **FixedLengthList**, found in the package moa.core. A FixedLengthList will grow to a maximum size. Beyond that, everytime a new element is added the oldest element is removed to make room.

MOA (Massive Online Analysis) [1] is a Java-based, open source framework for data stream mining. More details can be found on its [website](http://moa.cms.waikato.ac.nz/) and it can be found on GitHub as well (https://github.com/waikato/moa).

This specific code has a DOI: [![DOI](https://zenodo.org/badge/128976548.svg)](https://zenodo.org/badge/latestdoi/128976548)



REFERENCES

[1] A. Bifet, G. Holmes, R. Kirkby, and B. Pfahringer, “Moa: Massive online analysis,” J. Mach. Learn. Res., vol. 11, no. May, pp. 1601–1604, 2010.

[2] P. Kranen, I. Assent, C. Baldauf, and T. Seidl, “The ClusTree: indexing micro-clusters for anytime stream mining,” Knowl. Inf. Syst., vol. 29, no. 2, pp. 249–272, Nov. 2011.
