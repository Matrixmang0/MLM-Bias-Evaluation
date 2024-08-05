# Calculating Log Probability Score for Target-Attribute Association in BERT

This procedure describes how to calculate the log probability score, which measures the association between a target (person-denoting word) and an attribute (profession or emotion) in the BERT language model.

## Step-by-Step Procedure

### 1. Take a Sentence with a Target and Attribute Word

**Example:** "He is a kindergarten teacher."

- This is the starting point: a complete sentence containing both the target word ("He") and the attribute word ("kindergarten teacher").

### 2. Mask the Target Word

**Result:** "[MASK] is a kindergarten teacher."

- The target word (in this case, "He") is replaced with the `[MASK]` token. This prepares the sentence for obtaining the target probability.

### 3. Obtain the Probability of Target Word in the Sentence

**Calculation:** pT = P(he = [MASK]|sent)

- Use the BERT model to predict the probability of the target word ("he") appearing in the masked position, given the rest of the sentence.

### 4. Mask Both Target and Attribute Word. In Compounds, Mask Each Component Separately.

**Result:** "[MASK] is a [MASK] [MASK]."

- Both the target and attribute are masked. For compound words like "kindergarten teacher," each part is masked separately.

### 5. Obtain the Prior Probability

**Calculation:** pprior = P(he = [MASK]|masked sent)

- Calculate the probability of the target word ("he") appearing in the masked position when both the target and attribute are masked. This serves as a baseline probability without the influence of the attribute.

### 6. Calculate the Association

**Formula:** log(pT / pprior)

- The association is calculated by dividing the target probability (pT) by the prior probability (pprior) and then taking the natural logarithm of this ratio.

### Interpretation of the Log Probability Score

- **Positive Score:** The attribute increases the probability of the target word (positive association).
- **Negative Score:** The attribute decreases the probability of the target word (negative association).
- **Score Close to Zero:** Little to no association.

This method allows for quantifying gender bias in language models by measuring how different attributes (professions, emotions) affect the probability of gender-specific targets (he/she, man/woman, etc.).

## References

1. Bartl, M., Nissim, M., & Gatt, A. (2020). Unmasking Contextual Stereotypes: Measuring and Mitigating BERT's Gender Bias. In M. R. Costa-jussà, C. Hardmeier, W. Radford, & K. Webster (Eds.), Proceedings of the Second Workshop on Gender Bias in Natural Language Processing (pp. 1-16). Association for Computational Linguistics. [https://aclanthology.org/2020.gebnlp-1.1](https://aclanthology.org/2020.gebnlp-1.1)