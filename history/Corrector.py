import pycorrector

corrected_sent, detail = pycorrector.correct('韓元正在為您服務')
print(corrected_sent, detail)

# from pycorrector.macbert.macbert_corrector import MacBertCorrector

# nlp = MacBertCorrector("shibing624/macbert4csc-base-chinese").macbert_correct

# i = nlp('韓元正在為您服務')
# print(i)