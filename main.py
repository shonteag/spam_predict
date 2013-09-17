#!/usr/bin/env python3

import sys,time,math,operator;

#PUBLIC VARS and STRUCTS FOR VOCAB LIST GENERATION
vocab = {}; #vocab list of all words occuring in 4000 email training set. Form: {"word", # of containing emails}

#emails = [];

#PUBLIC VARS and STRUCTS FOR FEATURE ARRAYS
f_vectors = [];

#END PUBLIC VARS


#-------------------FILE SPLITTER---------------------------
def split_files():
    #create two new files, instead of destroying original
    file = open('data/spam_train.txt', 'r'); #original file, READONLY
    withhold = open('data/spam_train_validate.txt', 'w'); #new testing subdivision of training file (1000)
    newtrain = open('data/spam_train_use.txt', 'w'); #new training file (4000)
    
    email_counter = 1;
    
    if "-silent" not in sys.argv: print "Splitting data/spam_train.txt:";
    for line in file:
        if '0' or '1' in line: #0 or 1 being the "new email" marker
            if email_counter > 4000:
                withhold.write(line);
            else:
                email_counter=email_counter+1;
                newtrain.write(line);
        
    file.close();
    withhold.close();
    newtrain.close();
    
    if "-silent" not in sys.argv: print " ->Complete. New files _use.txt (4000) & _validate.txt (1000).";

#-------------------END FILE SPLITTER-----------------------
#-------------------VOCAB LIST GENERATION-------------------

#BEGIN FUNCTION
temp = {};
emails = [];

def parse_hashes(i, number): #adds temp_dict to vocab.
    #hash all emails individually for later looping.
    emails.append(temp.copy());
    
    if i <= number: #number = number of emails used to build vocab list
        #remove 0 and 1 from the temp
        if temp.get('0') != None: temp.pop('0');
        else: temp.pop('1');
        
        for word in temp:
            try:
                vocab[word] += 1;
            except KeyError:
                vocab[word] = 1;
    
    temp.clear();
    
#END FUNCTION

#BEGIN FUNCTION
def write_list_file():
    list_file = open('data/list_file.txt', 'w');
    
    if "-silent" not in sys.argv: print " ->Working on writing list file....";
    for key in vocab:
        list_file.write(key + " : " + str(vocab[key]) + '\n');
    
    f = open('data/hash.txt','w');
    for email in emails:
        f.write(str(email) + '\n');
    f.close();
    
    if "-silent" not in sys.argv: print " ->Complete.";
#END FUNCTION

#BEGIN FUNCTION
def drop_low_liers():
    if "-silent" not in sys.argv: print " ->Dumping low-lier occurence words";
    for key in vocab.keys(): #iterate over the vocab list (hashmap)
        if vocab[key] < 30: #if word appears in less than 30 distinct emails
            del vocab[key];
    if "-silent" not in sys.argv: print " ->Complete.";
#END FUNCTION

#BEGIN BUILD_VOCAB
#file is the file from which to pull raw data (_train.txt or test.txt)
#number is the number of emails used to build vocab list (4000 or 5000)
def build_vocab(file, number):
    print "Building vocab list using " + str(number) + " emails:";
    start_time = time.time();
    raw_data = open(file, 'r'); #training set of 4000 emails

    del emails[:];
    vocab.clear();

    holder = 1; #temp variable to make sure we dont parse empty hashtables
    if "-silent" not in sys.argv: print " ->Iterating over training emails...";
    i=0;
    for line in raw_data:
        #split the line into full words with the delimeter " " (space)
        thisline = line.split();
        
        for word in thisline:
            
            if temp.get(word) == None:
                temp[word] = 1;
        i += 1;
        parse_hashes(i, number);
    
    run = time.time() - start_time;
    if "-silent" not in sys.argv: print " ->Complete. " + str(run) + " seconds";
    #drop all entries with value less than 30.
    drop_low_liers();
    if "-silent" not in sys.argv: print "   ->" + str(len(vocab)) + " words in vocab list.";
    #write the vocab list file
    if "-loudfiles" in sys.argv: write_list_file();
    #return the list to the main program
    #return vocab;
    
    raw_data.close();
    return vocab,emails;
#END BUILD_VOCAB

#-------------------END VOCAB LIST GENERATION-------------------
#-------------------BUILD FEATURE VECTORS-----------------------

#BEGIN FUNCTION
def generate_vector_file():
    if "-loudfiles" in sys.argv: print " ->Working on writing feature vector file..."
    vfile = open('data/vector_file.txt', 'w');
    
    val = 1;
    for i in f_vectors:
        vec = f_vectors[val];
        vfile.write(vec);
        val += 1;
        
    if "-loudfiles" in sys.argv: print " ->Complete."
#END FUNCTION

#BEGIN BUILD_FEATURE_ARRAYS
#vocab_data is vocab dictionary built from specified training data
#emails_data is list of email hashes
def build_feature_arrays(vocab_data, emails_data):
    if "-silent" not in sys.argv:
        print "Creating feature vectors for " + str(len(emails_data)) + " emails..."
        print " ->Iterating over emails..."
    
    #clear old public variables and data structures
    f_vectors = []
    del f_vectors[:]
    f_vectors = [[] for i in range(0,len(emails_data))];

    start = time.time(); #run-time computation
    
    #start algorithm
    for index,email_hash in enumerate(emails_data): #number of hashed emails
        
        for key in vocab_data: #loop through all keys to check.
            if email_hash.get(key) != None: #if the word exists in the email hash
                f_vectors[index].append(1)
            else:
                f_vectors[index].append(0)
        
    #end algorithm
    

    fin = time.time() - start; #run-time computation
    
    if "-silent" not in sys.argv:
        print " ->Complete. " + str(fin) + " seconds";
        print "   ->" + str(len(f_vectors)) + " feature vectors generated of length " + str(len(f_vectors[1]));
    
    
    #if "-loudfiles" in sys.argv: generate_vector_file();
    return f_vectors;
#END BUILD_FEATURE_ARRAYS
    

#-------------------END BUILD FEATURE VECTORS-------------------
#-------------------PERCEPTRON STUFF----------------------------
learning_rate = .16;

#BEGIN DOT
def dot(vector1, vector2):
    return sum(val1*val2 for val1,val2 in zip(vector1,vector2));
#END DOT

#BEGIN CHECKER
def checker(vector, weights):
    result = 1 if (dot(vector,weights) >= .5) else 0 #rounding up or down to 0 or 1. better training.
    return result
#END CHECKER

#BEGIN PERCEPTRON_TRAIN
def perceptron_train(feature_vectors, email_data):
    print "Training perceptron..."
    print " ->Using " + str(len(feature_vectors)) + " feature vectors..."
    
    start = time.time(); #runtime computation
    
    weights = [float(0) for i in range(0,len(feature_vectors))]
    
    iterations = 0
    while True:
        error_count = 0;
        for index,vector in enumerate(feature_vectors):
            label = 1 if (email_data[index].get('0') == None) else 0;
            result = checker(vector,weights);
            error = label - result;
            if error != 0.0: #HAS TO BE 0.0 (float) OR IT NO WORKY!
                error_count += 1
                for index,value in enumerate(vector):
                    weights[index] += learning_rate * error * value
        iterations += 1
        
        #stop criteria
        if error_count <= 1 or iterations >= 15:
            break
                
        print "   ->" + str(iterations) + " iterations. " + str(error_count) + " errors."
    
    fin = time.time() - start;
    print " ->Complete. " + str(error_count) + " mistakes. " + str(iterations) + " iterations. " + str(fin) + " seconds."
    return weights,error_count,iterations;
#END PERCEPTRON_TRAIN

#BEGIN PERCEPTRON_TRAIN_AVG
def perceptron_train_avg(feature_vectors, email_data):
    print "Training perceptron average..."
    print " ->Using " + str(len(feature_vectors)) + " feature vectors..."
    
    start = time.time(); #runtime computation
    
    weights = [float(0) for i in range(0,len(feature_vectors))];
    
    avg_weights = [float(0) for i in range(0,len(feature_vectors))];
    counter = 0;
    
    iterations = 0
    while True:
        error_count = 0;
        for index,vector in enumerate(feature_vectors):
            label = 1 if (email_data[index].get('0') == None) else 0;
            result = checker(vector,weights);
            error = label - result;
            if error != 0.0: #HAS TO BE 0.0 (float) OR IT NO WORKY!
                error_count += 1
                for index,value in enumerate(vector):
                    weights[index] += learning_rate * error * value
            
            #do average thingymajiggy
            for index,value in enumerate(weights):
                avg_weights[index] += value;
            
            counter += 1;
               
        iterations += 1
        
        #stop criteria
        if error_count <= 1 or iterations >= 15:
            break
                
        print "   ->" + str(iterations) + " iterations. " + str(error_count) + " errors."
    
    print " ->Generating average weights..."
    for index,value in enumerate(avg_weights):
        avg_weights[index] = float(value) / float(counter);
    print " ->Complete. Returning average weights."
    
    
    fin = time.time() - start;
    print " ->Complete. " + str(error_count) + " mistakes. " + str(iterations) + " iterations. " + str(fin) + " seconds."
    return avg_weights,error_count,iterations;
#END PERCEPTRON_TRAIN_AVG

#BEGIN PERCEPTRON_TEST
def perceptron_test(weights, feature_vectors, email_data):
    print "Running test data..."
    print " ->Running on " + str(len(feature_vectors));
    
    start = time.time();
    
    num_errors = 0
    
    for index,vector in enumerate(feature_vectors):
        label = 1 if (email_data[index].get('0') == None) else 0;
        result = checker(vector,weights)
        if result == label:
            #correctly classified
            continue;
        else:
            num_errors += 1;
    
    fin = time.time() - start;
    
    misclassified = (float(num_errors) / float(len(feature_vectors)))
    print " ->Complete. " + str(fin) + " seconds."
    print "   ->Test Error: " + str(misclassified) + "; Misclassified " + str(num_errors) + " out of " + str(len(feature_vectors)) + " vectors.";
    return misclassified;
#END PERCEPTRON_TEST

#-------------------END PERCEPTRON STUFF------------------------
#-------------------BEGIN UTILITIES-----------------------------
def most_weighted(weights, vocab_data):
    a = [() for i in range(0,len(vocab_data))]
    
    for index,key in enumerate(vocab):
        a[index] = (key, weights[index]);
    
    a.sort(key=operator.itemgetter(1));
    
    print "\nMost negative words and weights:"
    for i in range(0,15):
        print a[i];

    print "\nMost positive words and weights:"
    for i in range(len(a)-15, len(a)):
        print a[i];
    


#-------------------END UTILITIES-------------------------------
#-------------------MAIN PROGRAM--------------------------------

#check arguments
if "-help" in sys.argv:
    print "Passable arguments : function";
    print "   -split    : Splits the data/spam_train.txt file into data/spam_train_use.txt (4000) and data/spam_train_validate.txt (1000)";
    print "   -loudfiles: Outputs vocab list and vector arrays to files";
    print "   -silent   : Supresses all command line program status print outs";
    print "   -validate : Run algorithm on split data file: data/spam_train.txt";
    print "   -test     : Run algorithm on test file: data/spam_test.txt";
    print "   -avg      : Run Average Perceptron Train algorithm"
    print "   -7        : Run algorithm to answer problem 7"
    sys.exit("Rerun the program without the -help argument.");

print "For list of command line arguments, -help"

#split files
if "-split" in sys.argv:
    split_files();
    
#PREPROCESSING

#----CONTROL SCRIPT FOR TRAINING/VALIDATION! NOT OFFICIAL TEST!
if ("-validate" in sys.argv):
    #training set of 4000 emails using validation of last 1000
    print "\nPreprocessing. 4000 emails for vocab list. 5000 feature vectors.";
    print "Using data/spam_train.txt";
    
    #generate vocab list and feature vectors
    spam_train_use_vocab,spam_train_emails = build_vocab('data/spam_train.txt', 4000);
    emails_use = spam_train_emails[0:4000];
    emails_validate = spam_train_emails[4000:5000];
    
    vectors = build_feature_arrays(spam_train_use_vocab, spam_train_emails);
    
    #split the returned feature vector into USE and VALIDATE
    vectors_use = vectors[0:4000];
    vectors_validate = vectors[4000:5000];
    
    #perceptron time, dude!!
    if "-avg"  in sys.argv:
        weights,mistakes,iterations = perceptron_train_avg(vectors_use,emails_use);
    else:
        weights,mistakes,iterations = perceptron_train(vectors_use,emails_use);
    test_error = perceptron_test(weights,vectors_validate,emails_validate);
    
    #utilities
    most_weighted(weights,spam_train_use_vocab)

#----COTRON SCRIPT FOR PROBLEM 7
elif "-7" in sys.argv:
    n = 100;
    spam_train_use_vocab,spam_train_emails = build_vocab('data/spam_train.txt', n);
    emails_use = spam_train_emails[0:n];
    emails_validate = spam_train_emails[4000:5000];
    
    vectors = build_feature_arrays(spam_train_use_vocab,spam_train_emails);
    vectors_use = vectors[0:n];
    vectors_validate = vectors[4000:5000];
    
    if "-avg"  in sys.argv:
        weights,mistakes,iterations = perceptron_train_avg(vectors_use,emails_use);
    else:
        weights,mistakes,iterations = perceptron_train(vectors_use,emails_use);
    
    test_error = perceptron_test(weights,vectors_validate,emails_validate);


#----CONTROL SCRIPT FOR ACTUAL DATA TEST!
elif "-test" in sys.argv:
    spam_train_vocab,spam_train_emails = build_vocab('data/spam_train.txt', 5000);
    train_vectors = build_feature_arrays(spam_train_vocab, spam_train_emails);
    
    if "-avg"  in sys.argv:    
        weights,mistakes,iterations = perceptron_train_avg(train_vectors, spam_train_emails);
    else:
        weights,mistakes,iterations = perceptron_train(train_vectors, spam_train_emails);
    
    a = spam_train_vocab.copy();
    #test file
    spam_test_vocab,spam_test_emails = build_vocab('data/spam_test.txt', 1000)    
    test_vectors = build_feature_arrays(a, spam_test_emails);
    perceptron_test(weights,test_vectors,spam_test_emails);

#-------------------END MAIN PROGRAM----------------------------
