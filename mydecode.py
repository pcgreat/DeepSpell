import pdb

from keras_spell import clean_text, PADDING, CONFIG, CharacterTable, read_top_chars
from keras.models import load_model
import numpy as np
model = load_model('data/wikivoyage/keras_spell_e24.h5')
ctable = CharacterTable(read_top_chars())


while True:
    question = input("Type a query:")
    question = clean_text(question.strip())
    question = question[:CONFIG.max_input_len] #TODO: for all length input
    question += PADDING * (CONFIG.max_input_len - len(question))
    question = question[::-1]

    print(question)

    X = np.zeros((1, CONFIG.max_input_len, ctable.size), dtype=np.bool)
    for i in range(1):
        for j, c in enumerate(question):
            try:
                X[i, j, ctable.char_indices[c]] = 1
            except KeyError:
                pass  # Padding

    res = model.predict(X)
    print("".join([ctable.chars[x] for x in np.argmax(res[0], axis=-1)]))
    print()

