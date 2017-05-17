import warnings
from asl_data import SinglesData
import math

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # test_sequences = list(test_set.get_all_Xlengths().values())
    # for current_sequence, current_length in  test_sequences:
    for word_id in range(0,len(test_set.get_all_Xlengths())):
#         = test_set.get_item_sequences(word_id);
        current_sequence, current_length = test_set.get_item_Xlengths(word_id)

        curr_iter_probabilities = {};
        for word, model in models.items():
            model_score = [];
            try:
                model_score = model.score(current_sequence, current_length)
            except:
                model_score = -math.inf;
                
            curr_iter_probabilities.update({word:model_score });

        guesses.append(max(curr_iter_probabilities, key=curr_iter_probabilities.get));
        probabilities.append(curr_iter_probabilities);

    return (probabilities, guesses);
