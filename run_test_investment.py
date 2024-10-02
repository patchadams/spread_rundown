import pandas as pd

def run_test_investment(df, buffer, bet_percentage):
    balance = 10000
    history = [10000]
    winners = 0
    losers = 0
    pushes = 0
    spread_buffer = buffer
    pred_winner = 0
    spread_field = 'spread_line'
    wrong = 0

    # Initialize a list to store "good bet" games
    good_bets = []

    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        bet = balance * bet_percentage  # Use the bet percentage provided
        is_away_good = row["predicted"] < row[spread_field] - spread_buffer
        is_home_good = row["predicted"] > row[spread_field] + spread_buffer
        result = None  # Initialize the result to None
        pick=''
        # Check if it's a good home or away bet and store the relevant information
        if is_away_good or is_home_good:
            balance -= bet
            losers += 1


            # Determine if the bet won or lost
            if is_away_good:
                pick='Bet Away'
                if row["home_score"] - row["away_score"] < row[spread_field]:
                    result = 'win'
                else:
                    result = 'lose'

            if is_home_good:
                pick='Bet Home'

                if row["home_score"] - row["away_score"] > row[spread_field]:
                    result = 'win'
                
                else:
                    result = 'lose'
            if row["home_score"] - row["away_score"] == row[spread_field]:
                    result = 'push'
                    balance += bet
                    pushes += 1

            # Append the result to the good_bets list
            good_bets.append({
                'game_id': row['game_id'],
                'home_away_diff': row['home_score'] - row['away_score'],
                'predicted': row['predicted'],
                'spread': row[spread_field],
                'action': pick,
                'result': result  # Add the win/lose result
            })

        if is_away_good:
            if row["home_score"] - row["away_score"] < row[spread_field]:
                balance += bet * (1.909)
                losers -= 1
                winners += 1

        if is_home_good:
            if row["home_score"] - row["away_score"] > row[spread_field]:
                balance += bet * (1.909)
                losers -= 1
                winners += 1


        # Count predicted winners and wrong predictions
        if row["home_score"] - row["away_score"] > 0 and row["predicted"] > 0:
            pred_winner += 1
        elif row["home_score"] - row["away_score"] < 0 and row["predicted"] < 0:
            pred_winner += 1
        else:
            wrong += 1

        history.append(balance)

    # Convert the good bets list to a DataFrame
    good_bets_df = pd.DataFrame(good_bets)

    return history, winners, losers, pushes, pred_winner, wrong, good_bets_df
