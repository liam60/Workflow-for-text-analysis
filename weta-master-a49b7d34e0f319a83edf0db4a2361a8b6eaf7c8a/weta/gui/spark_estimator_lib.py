

def run(inputs, settings):
    input_dataframe = inputs['DataFrame']
    params = settings['']
    estimator.setParams(**params)
    model = estimator.fit(input_dataframe)  # model
    output_dataframe = model.transform(input_dataframe)

    return {
        'DataFrame': output_dataframe,
        'Model': model
    }