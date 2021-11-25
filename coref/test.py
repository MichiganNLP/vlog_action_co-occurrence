# Load your usual SpaCy model (one of SpaCy English models)
import spacy
print("Spacy version: ", spacy.__version__)
nlp = spacy.load('en')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)

def test():
    doc = nlp(u'My sister has a dog. She loves him.')

    resolved_doc = doc._.coref_resolved
    print(resolved_doc)

    print(doc._.coref_clusters)
    print(doc._.coref_clusters[1].mentions)
    # doc._.coref_clusters[1].mentions
    # doc._.coref_clusters[1].mentions[-1]
    # doc._.coref_clusters[1].mentions[-1]._.coref_cluster.main


if __name__ == '__main__':
    test()
    