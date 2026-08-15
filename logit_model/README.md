# Card Pick Prediction Model

Predicts which card a player should pick from reward screens in Slay the Spire 2, using a **conditional logit** model with L1 regularisation.

## How it works

The model frames card selection as a discrete choice problem. Given a game state (deck, relics, HP, floor) and a set of offered cards (plus skip), it estimates the probability of each alternative being the best pick:

$$P(j \mid g) = \frac{e^{X_j \beta}}{\sum_{k \in g} e^{X_k \beta}}$$

Training uses **McFadden's trick** to reduce the conditional logit to standard binary logistic regression on pairwise feature differences, fitted via scikit-learn's `LogisticRegression` with L1 penalty.

### The core issue

After some experimentation, this is a dud. The model is not able to learn meaningful interactions between cards, since the parameter space is just too large. We have ~500 cards, and ~200 relics, so the interaction space is around 50,000 parameters. Even with tens of thousands of training examples, each card is seen relatively few times and the model fails to learn much about interactions between cards. Instead, it just ranks cards by their own merit, regardless of what else is present. In short, the model learns only which cards are generally good, not which cards are good for the given context.
