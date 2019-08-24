$('#keyword_button').click(function(){
    var key = $('#keyword').val();

    $.ajax({
        type:'POST',
        url:'get_tweet/',
        data:{ 'csrfmiddlewaretoken': $("input[name=csrfmiddlewaretoken]").val(),
                'keyword': key },
        success: ShowTweet,
        dataType: 'JSON'
    });

    function ShowTweet(data, textStatus, jqXHR)
    {
        document.getElementById('input').value = data.string;
    }
});


$('#Preprocessing').click(function(){
	var current = $('#input').val();
	// console.log('1');

	$.ajax({
			type: 'POST',
			url: 'next/1',
			success: Step2,
			data:{
				'input': current,
				'csrfmiddlewaretoken': $("input[name=csrfmiddlewaretoken]").val(),
			},
			dataType: 'JSON'
	});

	function Step2(data, textStatus, jqXHR){
		$('#results').append(data.msgs);
		current = data.current;
		// console.log('2');

		$.ajax({
			type: 'POST',
			url: 'next/2',
			success: Step3,
			data:{
				'input': current,
				'csrfmiddlewaretoken': $("input[name=csrfmiddlewaretoken]").val(),
			},
			dataType: 'JSON'
		});	

		function Step3(data, textStatus, jqXHR){
			$('#results').append(data.msgs);
			current = data.current;
			// console.log('3');

			$.ajax({
				type: 'POST',
				url: 'next/3',
				success: Step3,
				data:{
					'input': current,
					'csrfmiddlewaretoken': $("input[name=csrfmiddlewaretoken]").val(),
				},
				dataType: 'JSON'
			});

			function Step3(data, textStatus, jqXHR){
				$('#results').append(data.msgs);
				current = data.current;
				// console.log('4');

				$.ajax({
					type: 'POST',
					url: 'next/4',
					success: Step4,
					data:{
						'input': current,
						'csrfmiddlewaretoken': $("input[name=csrfmiddlewaretoken]").val(),
					},
					dataType: 'JSON'
				});

				function Step4(data, textStatus, jqXHR){
					$('#results').append(data.msgs);
					document.getElementById('Preprocessed').value = data.current;
				}
			}
		}	
	}
});


$('#Predict').click(function(){
    var current = $('#Preprocessed').val();
    if(!current.length){
        current = $('#input').val();
    }

	$.ajax({
		type: 'POST',
		url: 'next/5',
		success: Success1,
		data:{
			'input': current,
			'csrfmiddlewaretoken': $("input[name=csrfmiddlewaretoken]").val(),
		},
		dataType: 'JSON'
	});

	function Success1(data, textStatus, jqXHR){
		$('#results').append(data.prediction);

		$.ajax({
			type: 'POST', url: 'next/6', success: Success2, data:{ 'input': current,
				'csrfmiddlewaretoken': $("input[name=csrfmiddlewaretoken]").val(), }, dataType: 'JSON'
		});

		function Success2(data, textStatus, jqXHR){
			$('#results').append(data.prediction);

			$.ajax({
				type: 'POST', url: 'next/7', success: Success3, data:{ 'input': current,
					'csrfmiddlewaretoken': $("input[name=csrfmiddlewaretoken]").val(), }, dataType: 'JSON'
			});

			function Success3(data, textStatus, jqXHR){
				$('#results').append(data.prediction);
			}
		}
	}

});

function clear_all(){
    document.getElementById('results').innerHTML = '';
    document.getElementById('Preprocessed').value = '';
}

$('#random').click(function(){
    $.ajax({
        type:'POST',
        url:'get_random/',
        data:{ 'csrfmiddlewaretoken': $("input[name=csrfmiddlewaretoken]").val() },
        success: Append,
        dataType: 'JSON'
    });

    function Append(data, textStatus, jqXHR)
    {
        document.getElementById('input').value = data.string;
    }
});

$('#SVM').click(function(){
    $( '#CM_IMG' ).attr("src","/static/media/SVM_confusion_matrix.png");
    $('#accuracy').text('Training Accuracy: 81.54%');
    $('#SVM').addClass('active_tab');
    $('#NB').removeClass('active_tab');
    $('#RF').removeClass('active_tab');
    document.getElementById('modal-title').innerHTML = 'SVM';
    document.getElementById('definition_code').innerHTML = `def svmmodel(X,Y):    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)                
    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')
    text_clf_svm = pipeline.Pipeline([('vect', CountVectorizer()), ('tfidf', feature_extraction.text.TfidfTransformer()), ('clf-svm', SGDClassifier(loss='hinge', alpha=1e-3, max_iter=3, random_state=42)),])
    parameters_svm = {'vect__ngram_range':[(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf-svm__alpha': (1e-1, 1e-5),}
    gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=2)
    gs_clf_svm = gs_clf_svm.fit(X_train, Y_train)
    predicted = gs_clf_svm.predict(X_test)
    cm = metrics.confusion_matrix(Y_test,predicted)
                    
    # To display the graph                    
    np.set_printoptions(precision=2)
                    
    # Plot non-normalized confusion matrix
    plt.figure(figsize = (10,7))
    confusion_matrix_graph(cm,classes = Y_test, title='Confusion matrix, without normalization')
    plt.show()
                    
    # To display the scatter graph for predicted and tested values
    predic(Y_test, predicted)
    print(gs_clf_svm.best_score_)
                        
    print('....Saving!')
    joblib.dump(gs_clf_svm, 'resultssvm.pkl') 
    print('....Saved!')
                        
    res.append(gs_clf_svm.best_score_)
    return gs_clf_svm.best_score_`;
});

$('#NB').click(function(){
    $( '#CM_IMG' ).attr("src","/static/media/NB_confusion_matrix.png");
    $('#accuracy').text('Training Accuracy: 76.99%');
    $('#NB').addClass('active_tab');
    $('#RF').removeClass('active_tab');
    $('#SVM').removeClass('active_tab');
    document.getElementById('modal-title').innerHTML = 'Naive Bayes';
    document.getElementById('definition_code').innerHTML = `def naivebayesmodel(X,Y):
    # X=Reviews Y=labels
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=40)
    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')

    # Train and Test dataset size details
    print("X_train Shape :: ", X_train.shape)
    print("Y_train Shape :: ", Y_train.shape)
    print("X_test Shape :: ", X_test.shape)
    print("Y_test Shape :: ", Y_test.shape)

    clf=pipeline.Pipeline([
            ('tfidf_vectorizer', feature_extraction.text.TfidfVectorizer(lowercase=True)),
            ('nv_classifier', MultinomialNB(alpha=0.5, fit_prior=True))
        ])

    # clf = naiveBayes()
    trained = clf.fit(X_train, Y_train)
    print(trained)

    y_pred=clf.predict(X_test)
    print(y_pred)

    cm = metrics.confusion_matrix(Y_test,y_pred)
    
    #To display the graph   
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize = (10,7))
    #plot_confusion_matrix(cm,classes = Y_test, title='Confusion matrix, without normalization')
    confusion_matrix_graph(cm,classes = Y_test, title='Confusion matrix, without normalization')
    plt.show()
    
    #To display the scatter graph for predicted and tested values
    predic(Y_test, y_pred)
    
    acc = metrics.accuracy_score(Y_test,y_pred)
    print(acc)

    print('....Saving!')
    joblib.dump(trained, 'resultsnaivebayes.pkl') 
    print('....Saved!')
    
    res.append(acc)
    return acc`;
});

$('#RF').click(function(){
    $( '#CM_IMG' ).attr("src","/static/media/RF_confusion_matrix.png");
    $('#accuracy').text('Training Accuracy: 81.00%');
    $('#RF').addClass('active_tab');
    $('#NB').removeClass('active_tab');
    $('#SVM').removeClass('active_tab');
    document.getElementById('modal-title').innerHTML = 'Random Forest';
    document.getElementById('definition_code').innerHTML = `def randomforestmodel(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')

    clf=pipeline.Pipeline([
        ('tfidf_vectorizer', feature_extraction.text.TfidfVectorizer(lowercase=True)),
        ('rf_classifier', ensemble.RandomForestClassifier(n_estimators=300,verbose=1,n_jobs=2, max_features="sqrt",oob_score=True)),
        ('rf_classifier', ensemble.RandomForestClassifier(n_estimators=300,verbose=1,n_jobs=2, max_features="log2",oob_score=True)),
        ('rf_classifier', ensemble.RandomForestClassifier(n_estimators=300,verbose=1,n_jobs=2, max_features=None,oob_score=True))
    ])
    
    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in clf)
    # Range of 'n_estimators' values to explore.
    min_estimators = 15
    max_estimators = 300

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X, y)

            # Record the OOB error for each 'n_estimators=i' setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()

    y_pred=clf.predict(X_test)
    print(y_pred)

    cm = metrics.confusion_matrix(Y_test,y_pred)
    print(cm)
    
    #To display the graph 
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure(figsize = (10,7))
    confusion_matrix_graph(cm,classes = Y_test, title='Confusion matrix, without normalization')
    
    plt.show()
    
    #To display the scatter graph for predicted and tested values
    predic(Y_test, y_pred)
    acc = metrics.accuracy_score(Y_test,y_pred)
    print(acc)
    
    #save the model in the pickle file

    print('....Saving!')
    joblib.dump(trained, 'results.pkl') 
    print('....Saved!')
    
    res.append(acc)
    return acc`;
});