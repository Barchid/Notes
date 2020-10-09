# Pierre Falez thesis notes

# Chapter 4 : Frequency loss problem

The objective here is to design multi-layered SNNs in order to perform more complex tasks. Since the information in SNNs are transmitted thanks to spikes, it is necessary that the spiking activity remains high between neurons.  However having such SNNs can't be possible because of the **frequency loss problem**.

**Frequency loss problem definition :** the spiking activity is reduced as the layers are deep in the network. Consequently while the input frequency is high, the frequency remains much lower (or even null) after only two layers.

There are three solutions described to deal with the FLP.

## Section 4.1. : Mastering the FLP

### 4.1.1. Target frequency threshold
A target output frequency is explicitly specified and the objective of this method is to adapt the threshold of the neuron to reach the objective output frequency.

### 4.1.2. Binary coding


### 4.1.3. Mirrored STDP

    The way in which timestamps are generated with binary coding requires to take into account not only pre-synaptic spikes that occur before post-synaptic spikes, but all the spikes of the current input wave: pre-synaptic spikes that arrive shortly after the post-synaptic spike should contribute to the pattern.

**Why ?** : observed in vivo alongside STDP

# Chapter 5 : Comparison of the Features Learned with STDP and with AE
This chapter tries to answer 3 questions :

- How SNNs perform on complex datasets (color images, etc) such as CIFAR-100, etc.
- Perfomance gap between SNNs and standard methods ?
- What needs to be done to bridge this gap ?

Contributions of this chapter to answer the questons :

- Comparison of performances of pre-processing transformations to improve STDP training with color images.
- New threshold adaptation mechanism to learn patterns with temporal coding.
- Comparison between features learned by SNNs and sparse AE, an unsupervised standard approach

***Only single layer SNNs are evaluated*** because multi-layer SNNs are difficult to train because of the frequency loss problem.



## 5.1. Unsupervised Visual Feature Learning
*Defines the basis on which the features learnt by the SNNs will be compared with Sparse AE*

Introduces constraints : 
- STDP is a learning rule that is defined without an explicit objective function formulated, just like K-Means clustering.
- Properties of "good" visual features :
    - **Sparsity** : the extracted features must be sparse.
    - **Coherence of features** : features should be different in order to span the space of visual patterns. Try to have as little coherence as possible.

These properties will serve as a basis for the analysis of the tested feature extractor.

## 5.2. STDP-based Feature Learning
*Presents the new mechanisms*

Chapter 4 uses two neural coding :
- **Frequency coding** : problems due to frequency loss problem.
- **Binary coding** : loses information due to the binary nature of the coding. Workaround presented in chap. 4 (i.e. multiple repetitions) improves the performance but useless in complex images.

**Temporal coding** is used in Chap. 5. :
- Represents continuous values with one spike ==>
    - Avoid frequency loss problem
    - Avoid loss of information
- Sensitive to jitter noise :
    - offset of milliseconds impacts the represented value.
- Threshold is important in temporal coding :
    - Low threshold = neuron fires early = high value
    - High threshold = neuron fires lately = low value
    - Late firing neurons can integrate a larger part of the input spikes = can learn almost plain patterns

**Protocol :**
- Usage of IF neurons instead of LIF neurons
    - Uses less parameters = easier parameters search.
    - No issues from the leak :
        - balancing the leak so that the beginning of the pattern is remembered,
but also being able to forget the previous pattern between two samples
    - Membrane potentiel is forced to rest between each input.
- Usage of Multiplicative STDP
- WTA inhibition :
    - ensures neurons learn different patterns

### 5.2.1. Neuron threshold adaptation
*New threshold adaptation rule that works with temporal coding*

**Threshold** :
    - Influences the firing timestamps
    - ALlows to maintain homeostasis of the system.

**LAT : Standard adaptation threshold mechanism** :
    - Leaky adaptive threshold
    - Threshold of neuron increased after a spike in order to prevent it from firing too often
    - Exponential leak applied to help neurons with weak activities.
    - Search for suited values difficult because of two params required by the mechanism.
    - These params don't easily converge towards the types of patterns wanted (see paper Figure 5.1).

**Requirements for a new threshold adaptation mechanisme** :
    - Train the neuron to fire at an objective time ($t_{expected} \in [0, t_{exposition}]$)
    - Maintain the network's homeostasis
    - Firing timestamp t has to converge to $t_expected$

**Proposed threshold adaptation rule** :
    - First rule :
      - Each time neuron fires and each time it receives an inhibitory spike, threshold is adapted to reduce the difference between the actual time and the expected time t_expected.
      - All neurons in competition will apply the same change to their threshold to ensure the comptetition is not distorted.
      - *$t_{expected}$ requires exhaustive search*
    - Second rule :
      - Ensures the homeostasis = there is no winner that takes the advantage over the others because of WTA.
      - The winner increases its threshold and the other decrease a little
    - WTA inhibition : 
      - Only one neuron in the system is allowed to fire on each sample
      - Only one neuron apply STDP per sample.
    - Side notes :
      - Threshold must be initialized with small values to ensure neural activity.
      - With high thresholds, there is no neuronal activity and thus no learning (and no threshold adaptation)

### 5.2.2. Output conversion function
*Introduction of conversion from spike to numerical values in order to be used by traditionnal classifier*

The conversion function creates a feature vector $\bold{g}$ that consists of numerical values $g_i = f_{out}(t_i)$.

Basically, the conversion function is an inverse function to translate the temporal coding. It returns a high value when there is an early spike.


### 5.2.3. On/Off filters
*Introduction of preprocessing steps for color images*

- Preprocessing is required to transform an image into spikes.
- Preprocessing must help the STDP to learn correlations between spikes.
- **DoG/Gabor filter** : extract the edges of a grayscale image. Usually used.

**First strategy : RGB color opponent channels** : the coding is applied to channels computed as differences of pairs of RGB channels (red-green, green-blue and blue-red).

**Second strategy : inspired from biology** :
- three channels exist in the lateral geniculate nucleus :
  - black-white opponent channel (corresponds to grayscale image)
  - red-green opponent channel
  - yellow-blue opponent channel
- Applies on-center/off-center coding to the red-green and yellow-blue channels
  - $0.5 \times R + 0.5 \times G - B$

**Four configurations** possible :
- grayscale only
- RGB opponent channels
- Bio-color
- Grayscale + Bio-color


## 5.3. Learning visual features with sparse auto-encoders
*Explains the visual feature learned by a sparse auto-encoder*

**Sparse auto-encoders** :
- neural networks for unsupervised learning that find latent representation to reconstruct the input image.
- Here, only single-layer AE are used because :
  - comparison with one-layer SNN and multi-layer AE won't be fair
  - multi-layer SNN are just emerging so it won't be fair to use them.
- Organize as follows :
  - An encoder : maps an input image into a latent representation
  - A decoder : maps a latent representation to an output image that must be the same as the input.
- Avoid trivial solution (e.g. identity function)
  - Many techniques available (*weight regularization, explicit sparsity constraitnts, regularization of the Jacobian of the encoder output*). Or change the objective function from reconstruction to another criterion.

- Auto-encoder used here :
  - Single-layer
    - The output of the encoder (= latent representation) is the learnt visual features.
  - Type of auto-encoder used : **Sparse** auto-encoder with :
    - L2 weight regularization
    - Defined Sparsity term from a paper...

## 5.4. Experiments

### 5.4.1. Experimental protocol
- SNN and AE are single layered with $n_{features}$ hidden units

- **Feature learning** : $X_{train} = (X_1, X_2, ..., X_n)$ is the dataset where $n_{patches}$ batches are randomly sampled. The batches are fed to the feature learning algorithm
- **Image recognition** : the features extracted from SNN/AE are fed into a classifier (SVM).

### 5.4.2. Datasets
Three datasets used (their grayscale versions are also used):
- CIFAR-10
- CIFER-100
- STL-10

### 5.4.3. Implementation details
.......

### 5.4.4. Color processing with SNNs
The four strategies discussed in sec [link](#523-onoff-filters) are compared with SNNs.

**Results :**
- Bio-color and RGB opponent channels have similar recognition rates
- Grayscale is better (counter-intuitive)
  - On-center/off-center coding should be the source of this information loss.
  - But this coding is required to extract edges and feed SNN with spike trains that represent specific visual information.
  - Can't use directly the RGB images because the mechanisms are not adapted to learn with this type of data.
    - Threshold adaptation rule not effective
      - The sum of input patterns vary from dark patches (close to 0) to blank patches (sum can go high)
    - Many filters found with training from raw RGB consists of dead or repeated features. ==> **Uninformative**
- Grayscale + color is the best !
  - DoG filtered color images contain information that DoG-filtered grayscale images doesn't
  - **Grayscale + color is used for the rest of the chapter**

### 5.4.5. SNNs versus AEs
AEs perform consistently better than SSs (even with the use of a one-layer AE).

**WHY ?** :
    - Color provides better results (can be explained by the significance of the color in object recognition). The SNNs cannot handle color well (at least not with color coding).
      - **Need for more efficient neural coding to handle color well.**

**Comparison of filters** :
- SNN filters are mostly edges and some blobs with one or two dominant colors.
  - Well defined features, simple
  - Can easily be understood
  - Correspond to biological observations
  - Not effective in practice
  - **Explanation** :
    - Pre-processing (on-center/off-center coding) highlights edges !!
    - STDP rules : once a given unit has learned a pattern, repeated expositions to this pattern will reinforce the sensitivity to this pattern unitl the weights reach 1 or 0
      - STDP leads to saturated weights
    - SNNs do not raise dead units (features stuck in a state where there is no significant pattern). Can be explained by :
      - Lateral inhibition (prevents neuron from learning similar patterns)
      - Saturated regime of STDP
- AE learn more complex features (edges and blobs can be observed but include larger range of color or gray levels).

## 5.5. Result analysis and properties of the networks

### 5.5.1. On-center/off-center coding

*Investigates the impact of on/off-center coding on classification*

This coding impacts the visual features learned bu STDP but does it impact the accuracy ?

**Two systems are compared, each w/ and w/o preprocessing**
1. AE (same protocole) 
2. SVM performing directly on images

- Using on/off-center coding decreases the accuracy of the classification. 
  - Confirms this coding decreases the performance of SNNs. Extracting edges with DoG selects only a subrange of spatial frequencies.
- filtered Color AE is on par with grayscale
- filtered color SVM is worse than grayscale
- On-center/off-center coding can't handle color images
  - Edges are already effectively represented in grayscale
  - Additionnal information of color is in uniform image regions
- SOTA unsupervised SNN are only investigated on MNIST, which is just composed of edges
- Must find a way to process raw RGB but it is not working now (see above)

### 5.5.2. Sparsity
- Compare AE and SNN in term of sparsity
- SNNs are sparser than AE
  - SNNs relies on lateral inhibition --> leading to versy sparve activations of the features.
- Sparsity is generally correlated with accuracy
- Too much sparsity is not good for AE

Is sparseness is an issue for SNNs ?
- Experiments where lateral inhibition is disabled is run
  - SParsity decreased and accuracy too.
  - Sparsity is desirable for good representation but excessive sparsity is detrimental for accuracy

### 5.5.3. Coherence
*Comparison between AE and SNN in term of incoherence quality*

Incoherence is a measure of quality of the learned features : one feature should not be obtained by a linear combination of other features.

- Incoherence low
  - features are redundant
  - Can negatively impact accuracy
  - Redundant features can overweight other features

- **Comparison**
  - STDP-based SNNs product more coherent features
    - Can explain their lower performance
    - Smaller variety of filters
  - Pairwise coherence between two SNN features is higher
    - SNNs can learn almost identical features
  - **Conclusion :** WTA inhibition fails to prevent neurons to react to the same patterns. Must design more effective inhibition mechanisms.

### 5.5.4. Objective function
*Tries to identify the criteria that are optimized by STDP rules in order to understand the related learning process. Does STDP aims to optimize the reconstruction criteria ? If not, this shows that STDP is not suited for reconstruction*

- Objective function of STDP is not explicitly expressed
- Objective function of AE is minimizing reconstruction error

**--> Are features learned by STDP are suited for reconstruction ?**

- How to check that ?
  - Reconstruct the test image from the visual features
    - In AE, this is the decoder's job
    - In SNNs, patches are reconstructed as a linear combination of the filters weighted by their activations for the current sample. Images are reconstructed from patches by averaging the values of overlapping patches at each location
- Reconstruction error is much higher for SNNs than AEs
    - STDP does not allow reconstruction
    - Edges are reconstructed with less details than orgininal image
      - Can be explained by learned features that are much more elementary and sparse (no complex patterns)
    - Global illumination degraded
- Explanation of results
  - SNN only process DoG-filtered images
  - Reonstruction error can be much lower if able to process raw images directly 

**Conclusion :** 
- SNNs learn to reconstruct images (although this is not explicit). 
- Minimizing reconstruction error is not sufficient to provide meaningful representations.
  - Recent AE include additional criteria to ensure good representation
  - Must work on these criteria in STDP
- STDP can behave similarly to ICA or PCA (dimension reduction)


### 5.5.5. Using Whitening Transformations with Spiking Neural Networks

- On-off filtering is an unsatisfactory solution
  - Due to information loss (essentially the edges are used)
- Solution : **whitening**
  - Centered, normalized and decorrelated data
  - Showed improved results in traditionnal methods
- Whitened data converted into spikes does'nt allow to learn effective features
  - So it shows very low accuracy
- Apply whitening transformation is computationally expensive (and not easily implementable on neuromorphic architectures)
- Solution with a kernel etc... (TODO : work presented by Tirilly to read !)

### 5.6. Conclusion
- STDP-based SNNs unable to deal with RGB images naturally. Need for preprocessing
- Common on-center/off-center image coding usedi in SNNs result in an information loss (only edges are taken into account)
- WTA inhibition results in overly sparse features and does not prevent neurons from learning the same features in practice
- STDP-based learning rules produce features that enables reconstruction but this is not explicitly optimized for this. Qualty of reconstruction is harmed by the model (preprocessing, etc)

Whitening is a solution to replace on/off filtering. But need to provide whitening-like mechanisms that can be used in neuromorphic architectures.

# Chapter 6. Training Multi-layer SNNs with STDP and Threshold Adaptation
*Extends previous mechanisms to allow learning with multi-layered SNNs*

Contributions of chap. 6
- Threshold adaptation mechanism
- Protocol to train multi-layer network
- Experimentation to evaluate the impacts of
  - Threshold adaptation
  - Inhibition mechanism
  - STDP rule
- Test the combination of multiple networks trained with =/= parameters to improve the classification rate

## 6.1. Network architecture
*Describes the general details of the designed SNN*

- Composed of Feed-Forward layers.
- IF neurons are used (reduce the number of params compared to LIF). 
- on/off filtering preprocessing
- Grayscale images used
- Temporal coding to convert from images to spikes
- Types of layers
  - Convolution
  - Pooling
    - Synaptic weightand threshold are 1
    - A pooling neuron directly fires a spike when it receives a spike from its receptive field
      - Mimics max pooling
  - Fully-connected

## 6.2. Training multi-layered Spiking Neural Networks
*Describes the model*
- Same mechanisms used as in the previous chapter
- No delay used in the network
  - For simplification
  - Increase parallelism
  - Reduce parameters number
  - But delay may play a major role in the learning of temporal patterns

### 6.2.1. Threshold adaptation rule
*Extends the threshold adaptation rule*

- Reuse threshold adaptation rule of [Secion 5.2.1](#521-neuron-threshold-adaptation).
- Add new parameter that limits the minimum value that the threshold can take.
  - Forces the neurons to integrate a minimum umber of spikes
  - Reinforces the connections
  - Useful to ensure that neurons learn effective patterns
- Remove WTA inhibition during the inference stage
  - WTA inhibition reduces the spiking activity and, in practice, does not prevent from learning redundant patterns.
  - Investigate soft inhibition
    - Uses inhibition spikes (reduces the membrane voltage of other neurons by a constant)


### 6.2.2. Network output
*Offers a spike-to-value conversion function to interpret the output of the network*

The new method :
- Takes $t_{expected}$ into account
- Latency coding 
  - Early output spike = high value


### 6.2.3. Training
*Describes the protocol used to train multi-layered SNNs*

Convolution requires to perform non-local operations and to use non-local memory since they use shared weights.

To reduce the cost of global communication needed by convolutions, a specific protocol is used :
- One ayer is trained at a time (from the 1st layer to the last)
- During the training of a convolution layer, only one column is activated to discard the usage of inter-column communications.
  - Necessary since pooling layers require the same filters in adjacent columns
- Once the layer is trained, its parameters (weights and thresholds) are fixed and are copied onto the other columns of the layer.

## 6.3. Results
### 6.3.1. Experimental protocol
- Adapted protocol of [Section 5.4](#541-experimental-protocol)
- For each trained layer, the training set is processed $n_{epoche}$ times
- Learning rate decay
  - Converge to a stable state
- After training, training and test sets are processed by the network to convert all the samples into their output representation
- Must produce a feature vector 
  - If output has multiple columns, sum-pooling is applied
- SVM classifier
- Classification rate is averaged over 10 runs
- Sparsity of output vector is computed with the equation in [Section 5.5.2](#552-sparsity)

### 6.3.2. MNIST
#### Threshold target time

- Impact of the parameter $t_{expected}$ studied.
  - Directly impact learned filters and classification perfs
    - Low values lead to very local patterns
    - Large values lead to more global patterns
  - Large value makes integrate a large number of spikes
    - Better accuracy
  - Very late $t_{expected}$ is harmful though
    - Latest spikes are not useful for classification
  - Equation 5.3 is disabled lead to homeostais lost
    - Lower accuracy
  - Different $t_{expected}$ across the layers is bad
    - Neurons on prev layer are trained to fire at a defined $t_{expected}$
      - If next layer has a lower $t_{expected}$, it misses some spikes of the previous layer
      - If next layer has a higher $t_{expected}$ it integrates spikes which arise too late ==> the patterns is not similar to those recognized by the input neurons
  - With a little difference of $t_{expected}$ between layers, the performance remain stable
    - Noise resistant
  - Best results are done with a very little offset
    - Seems to reinforce resistance to the noise without integrating unrelated spikes
  - **Conclusion :**
    - Use a single value for $t_{expected}$ is enough for a multi-layered SNN

#### Inhibition
*Show the impact of inhibition policies*

Three inhibition policies are compared
1.  WTA
2.  Soft inhibition
3.  No inhibition

Observations :
- Increasing the hardness of inhibition during inference decreases the accuracy
- The effect of inibition is accuentuated after each layer
  - Impacts sparsity and recongition rate n the FC layer
  - Effect visible with soft inhibition but maximal in WTA
  - Sparsity of FC layer is 1 and the accuracy is only 63%
- Higher level of activity helps to learn better rep

#### STDP rule

Three STDP rule compared :
1. Additive STDP
2. Multiplicative STDP
3. Biological STDP

- Additive STDP :
  - Baseline perf of 96%
  - Relatively high level of sparsity
  - Saturation effect = binary weights (close to 0 or 1)
- Multiplicative STDP
  - Reduces saturation thanks to param $\beta$
    - Big value of $\beta$ greatly decreases number of binary weights
    - Best perf is 99.22%
  - Biological STDP
    - Best of 98.47%

- Filters learned by biological STDP look different than the others
- Mult & add STDP rules never learn patterns that overlap on the on and off channels
- Bio STDP :
  - Leads to filters with reinforced connection on the two channels (red & green)
- Filters learned by add & multi STDP lead to identifiable digits
- Bio-STDP leads to filters that are less clear.
  - See figure 6.5
  - The non-linearity brought by biological STDP allows learning more complex features

#### Multiple Target Timestamp Networks
*Investigation of networks that contain several groups of neuron with different $t_{expected}$*

- Representations learned with != $t_{expected}$ can obtain != patterns
  - can be useful to the classifier
- Protocol :
  - $n$ networks are trained independantly with a given $t_{expected}$
  - Concatenate the output features of each network to create a big feature vector $g$
- Using multiple targets improve the classification perf
- Network reaches a recognition rate of 98.6% (better than existing methods)
- Explaination
  - Combining different $t_{expected}$ allows detecting more varied patterns

### 6.3.3. Faces/Motorbikes
Use the model in Faces/Motorbikes dataset to see if the model performs well in more realistic images.

**Protocol :**
- Extract two classes from Caltech-101 (faces and motorbikes)
- Resize to 250x160px
- Convert into grayscale
- Training set = 474 samples
- Test set = 759 sample 
- Similar results of the other SNN of SOTA


## 6.4. Discussion
Convolution in SNNs is an issue for multi-layered SNNs. Convolution columns are trained independantly but necessary to copy the weight and threshold values to the other columns to mimics the weight sharing mechanism.

- Fully hardware-implementable SNN using bio-inspired classifiers is unescapable
- Reward STDP : reinforcement learning STDP to make multilayered SNNs working with Spiking classifier
- $t_{expected}$ is a parameter with strong impact on the classif perf
  - Interesting to introduce an auto-adaptable version of this parameter so that neurons can find by themselves the best timing for firing

## 6.5. Conclusion
- Previous multi-layered SNNs require exhaustive search to optimize params
- THreshold adaptation mechanism uses a single parameter for all layers and allows varied patterns
- Removing inhibition during inference step helps to reduce sparsity of model activity which leads to an improvement of perf
- Bio-STDP rule helps improving the network perf by introducing non-linearity


# Chapter 7. Conclusion and future work
## 7.1. Conclusion
- Limitation of SNN = poor perf compared to deep learning
- SNNs can't be used for complex cv tasks
- Work of the thesis
  - Improve classification tasks for SNN
  - Focus on STDP
  - Must be compatible with neuromorphic hardware
  - Multi-layered SNNs
- **First contribution**
  - Dev of SNN simulators 
  - N2S3
    - = Akka simulator
    - Easy to change to make quick tests
    - Too heavy to simulate large networks
    - Synchronization bottleneck
  - CSNNS
    - Efficient for IF neurons w/ temporal coding
- **Second contribution** : Frequency loss problem
  - Prevent the use of multi-layer SNN
  - Activity across layers drops drastically
  - Mechanism to deal with it
    - Target Frequency Threshold (TFT)
      - Threshold adaptation mechanism
      - Train the neuron to reach desired output frequency
    - Binary coding
      - Convert images into spike trains in order to prevent the loss of frequency
    - Mirrored STDP 
      - STDP rule that exploit binary coding to avoid FLP
  - Results :
    - Avoid FLP and maintain good classification score
    - Binary coding loses information on the conversion process
- **Third contribution** :
  - Threshold adaptation rule to allow STDP to learn patterns on samples converted with latency coding
  - Test behaviour of SNNs with RGB images
    - Adapt on/off filtering policy (used in grayscale) for RGB images
  - Comparison with sparse AE
    - Sparsity,accuracy, filter coherence, reconstruction error
  - Results
    - WTA inhibition lead to inefficient representations
    - On/Off filtering = loss of information = perf decrease
  - Whitening
    - Does not retain a specific frequency
    - Good results
- **Fourth contribution**
  - Set up multi-layered SNN trained with STDP
  - Adaptation of previous designed Threshold adaptation rule for better control
  - Protocol to train multi-layered SNN
  - Test of mechanisms :
    - STDP rule
      - bio-STDP better than multiplicative & additive STDP
      - Non-linearity of bio-STDP
    - Inhibition system
      - Removing the inhibition after training allows to reduce the sparsity and increase accuracy
    - Threshold adaptation
      - All thresholds can be optimized with a single parameter $t_{expected}$ & it can control the type of pattern learned by the neurons


## 7.2. Future work
Three ideas to continue :
- Improve simulation of SNNs
- Enhance the perf of spiking models
- Make models fully compatible with hardware implementation

### 7.2.1.
- To bridge the gap between SNN and ANN, we need to deal with SOTA datasets (ImageNet, COCO, etc).
- Processing datasets like that means big-sized networks
  - Simulators must be able to handle big networks in a decent amount of time
- Computations must be parallelized
  - Apply same instructin on multiple data (SIMD)
  - or simultaneously executing different operations (MIMD)
- SIMD is not possible for SNN because of spatial and temporal sparsity
  - SIMD makes all the data updated at the same time
- Works in progress to make SNN simulation possible on GPU
- MIMD use
  - Challenge is the synchronization between the // units
  - units must be temporally coherent
- N2S3 does'nt work
  - Inefficient to simulate large networks
  - Main bottleneck is the global synchronizer

### 7.2.2. Improving the learning in siking neural networks
Mechanisms must e studied further to improve perf of these models

- Work of spike frequency must be continued
- What's necessary
  - Maintain sufficient activity
  - more work on inhibition systems
    - Prevent from learning same patterns
    - BUT let enough spikes pass trhough
  - Better preprocessing than on/off filtering
    - Whitening
  - Improve STDP
    - Add something other than forward connections 

### 7.2.3. Hardware implementation of Spiking Neural Networks
Multi-layered SNNs trained with STDP not fully available on neuromo hw.
- These models must be compatible with dedicated architectures in order to take advantage of their energy efficiency

Three major mechanisms not implemented in HW :
1. Pre-processing
2. Shared filters in convolution layers
3. Classifier

- Pooling layers are useful
  - Reduce dimension data
  - Position invariance improved
  - Difficult to copy weight of filters
- Classifier problems
  - Supervised STDP : teach signal to force neuron to learn desired patterns
  - Forcing output neurons to spike
    - Reinforce good connections
  - Anti-STDP
    - Force non-correct output neurons to not fire
      - Can't prove if 
  - Reward STP
    - Modulate each synaptic weight update according to a reward factor
      - Must find a way to propagate reward in spikes (as for BP)
