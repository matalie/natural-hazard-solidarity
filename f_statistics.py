def calc_IRR(df):
    responses = df['respondent_id'].nunique()
    
    # get inconsistent answers (options were switched in first and last task of surveys thus !=)
    same_choice = df[(df['choice_first'] != df['choice_last'])]
    num_same_choice = len(same_choice)
    
    # get consistent answers
    different_choice = df[(df['choice_first'] == df['choice_last'])]
    num_different_choice = len(different_choice)
    
    # calculate IRR
    IRR_choice = (num_same_choice) / (num_same_choice + num_different_choice)
    print(f"IRR: {IRR_choice}")
    
    sqer_IRR_choice = np.sqrt((IRR_choice * (1 - IRR_choice)) / responses)
    z_crit = norm.ppf(0.975)  # 95% confidence interval
    CI_plus = IRR_choice + (z_crit * sqer_IRR_choice)
    CI_minus = IRR_choice - (z_crit * sqer_IRR_choice)
    
    print(f"CI Plus: {CI_plus}")
    print(f"CI Minus: {CI_minus}")

    # swap error
    swap_error_choice = (1 - np.sqrt(1 - (2 * (1 - IRR_choice)))) / 2
    print(f"Swap Error Choice: {swap_error_choice}")