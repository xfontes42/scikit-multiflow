import copy as cp
import numpy as np
from skmultiflow.core import BaseSKMObject, ClassifierMixin, MetaEstimatorMixin
from skmultiflow.bayes import NaiveBayes


class HoeffdingEnsembleClassifier(BaseSKMObject, ClassifierMixin, MetaEstimatorMixin):
    """ Hoeffding ensemble classifier.

    Parameters
    ----------
    n_estimators: int (default=5)
        Maximum number of estimators to hold.
    base_estimator: skmultiflow.core.BaseSKMObject or sklearn.BaseEstimator (default=NaiveBayes)
        Each member of the ensemble is an instance of the base estimator.
    alpha: float (default=)
    beta: float (default=0.8)
        Factor for which to decrease weights by.
    gamma: float (default=0.1)
        Weight of new experts in ratio to total ensemble weight.


    Notes
    -----
    The Hoeffding Ensemble Classifier (HCEnsemble) is a novel method for using any online learner based
    on AddExp [1]_ and DWM [2]_ for drifting concepts. It uses the Hoeffding bound as a way to define when it's
    time to remove experts or which expert to remove.

    References
    ----------
    .. [1] Kolter and Maloof. Using additive expert ensembles to cope with Concept drift.
       Proc. 22 International Conference on Machine Learning, 2005.

    .. [2] Kolter and Maloof. Dynamic weighted majority: An ensemble method
       for drifting concepts. The Journal of Machine Learning Research,
       8:2755-2790, December 2007. ISSN 1532-4435.
    """

    class WeightedExpert:
        """
        Wrapper that includes an estimator and its weight.

        Parameters
        ----------
        estimator: StreamModel or sklearn.BaseEstimator
            The estimator to wrap.
        weight: float
            The estimator's weight.
        """
        def __init__(self, estimator, weight):
            self.estimator = estimator
            self.weight = weight
            self.seen = 0
            self.incorrect = 0

        def expert_loss(self):
            return self.incorrect / self.seen if self.seen > 0 else 0

    def __init__(self, n_estimators=5, base_estimator=NaiveBayes(), beta=0.8,
                 gamma=0.1, hoeffding_confidence=0.95, hoeffding_min=1):
        """
        Creates a new instance of HCEnsemble.
        """
        super().__init__()

        self.n_estimators = n_estimators
        self.base_estimator = base_estimator

        self.beta = beta
        self.gamma = gamma
        self.hoeffding_confidence = hoeffding_confidence
        self.hoeffding_min = hoeffding_min

        # Following attributes are set later
        self.epochs = None
        self.num_classes = None
        self.experts = None

        self.reset()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fits the model on the supplied X and y matrices.

        Since it's an ensemble learner, if X and y matrix of more than one
        sample are passed, the algorithm will partial fit the model one sample
        at a time.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Features matrix used for partially updating the model.

        y: Array-like
            An array-like of all the class labels for the samples in X.

        classes: numpy.ndarray (default=None)
            Array with all possible/known class labels.

        sample_weight: Not used (default=None)

        Returns
        -------
        AdditiveExpertEnsemble
            self
        """
        for i in range(len(X)):
            self.fit_single_sample(X[i:i+1, :], y[i:i+1], classes, sample_weight)
        return self

    def predict(self, X):
        """ Predicts the class labels of X in a general classification setting.

        The predict function will take an average of the predictions of its
        learners, weighted by their respective weights, and return the most
        likely class.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            A matrix of the samples we want to predict.

        Returns
        -------
        numpy.ndarray
            A numpy.ndarray with the label prediction for all the samples in X.
        """
        preds = np.array([np.array(exp.estimator.predict(X)) * exp.weight
                          for exp in self.experts])
        sum_weights = sum(exp.weight for exp in self.experts)
        aggregate = np.sum(preds / sum_weights, axis=0)
        return (aggregate + 0.5).astype(int)    # Round to nearest int

    def predict_proba(self, X):
        """ Not implemented for this method.
        """
        raise NotImplementedError

    def fit_single_sample(self, X, y, classes=None, sample_weight=None):
        """
        Predict + update weights + modify experts + train on new sample.
        """
        self.epochs += 1
        self.num_classes = max(
            len(classes) if classes is not None else 0,
            (int(np.max(y)) + 1), self.num_classes)

        # Get expert predictions and aggregate in y_hat
        predictions = np.zeros((self.num_classes,))
        for exp in self.experts:
            exp.seen += 1
            y_hat = exp.estimator.predict(X)
            predictions[y_hat] += exp.weight
            if np.any(y_hat != y):
                exp.incorrect += 1
                exp.weight *= self.beta

        # Output prediction
        y_hat = np.array([np.argmax(predictions)])

        # Check hoeffding bound
        if len(self.experts) > 1:
            self.experts = sorted(self.experts, key=lambda _exp: _exp.expert_loss(), reverse=True)
            n = min(self.experts[0].seen, self.experts[1].seen)
            if n > self.hoeffding_min:
                epsilon = self.compute_hoeffding_bound(range_val=1.0, confidence=self.hoeffding_confidence, n=n)
                if self.experts[1].expert_loss() - self.experts[0].expert_loss() > epsilon:
                    self.experts.pop(0)

        # If y_hat != y_true, then add a new expert
        if np.any(y_hat != y):
            ensemble_weight = sum(exp.weight for exp in self.experts)
            new_exp = self._construct_new_expert(ensemble_weight * self.gamma)
            self.experts.append(new_exp)

        # Pruning to self.n_estimators if needed
        if len(self.experts) > self.n_estimators:
            self.experts.pop(0)

        # Train each expert on X
        for exp in self.experts:
            exp.estimator.partial_fit(X, y, classes=classes, sample_weight=sample_weight)

        # Normalize weights (if not will tend to infinity)
        if self.epochs % 100:
            ensemble_weight = sum(exp.weight for exp in self.experts)
            for exp in self.experts:
                exp.weight /= ensemble_weight

    def get_expert_predictions(self, X):
        """
        Returns predictions of each class for each expert.
        In shape: (n_experts,)
        """
        return [exp.estimator.predict(X) for exp in self.experts]

    def _construct_new_expert(self, weight=1):
        """
        Constructs a new WeightedExpert from the provided base_estimator.
        """
        return self.WeightedExpert(cp.deepcopy(self.base_estimator), weight)

    def reset(self):
        self.epochs = 0
        self.num_classes = 2
        self.experts = [self._construct_new_expert()]

    @staticmethod
    def compute_hoeffding_bound(range_val, confidence, n):
        r""" Compute the Hoeffding bound, used to decide how many samples are necessary at each node.

        Notes
        -----
        The Hoeffding bound is defined as:

        .. math::

           \epsilon = \sqrt{\frac{R^2\ln(1/\delta))}{2n}}

        where:

        :math:`\epsilon`: Hoeffding bound.

        :math:`R`: Range of a random variable. For a probability the range is 1, and for an information gain the range
        is log *c*, where *c* is the number of classes.

        :math:`\delta`: Confidence. 1 minus the desired probability of choosing the correct attribute at any given node.

        :math:`n`: Number of samples.

        Parameters
        ----------
        range_val: float
            Range value.
        confidence: float
            Confidence of choosing the correct attribute.
        n: int or float
            Number of samples.

        Returns
        -------
        float
            The Hoeffding bound.

        """
        return np.sqrt((range_val * range_val * np.log(1.0 / confidence)) / (2.0 * n))
