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
