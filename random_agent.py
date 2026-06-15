import random
from snake_game import SnakeGameAI

def play_random():
    game = SnakeGameAI()
    score_sum = 0
    games = 0
    
    while games < 10: # Play 10 random games
        # get random move
        move = random.randint(0, 2)
        final_move = [0, 0, 0]
        final_move[move] = 1

        # perform move
        reward, done, score = game.play_step(final_move)

        if done:
            games += 1
            print(f'Game {games} finished with Score: {score}')
            score_sum += score
            game.reset()

    print(f'Average score over 10 games: {score_sum/10}')

if __name__ == '__main__':
    print("Agentul random începe să joace...")
    play_random()
