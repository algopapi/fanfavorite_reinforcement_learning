from matplotlib import pylab

pylab.figure(figsize=(18, 9))
def PlotModel(agent, score, episode):
    agent.scores.append(score)
    agent.episodes.append(episode)
    agent.average.append(sum(agent.scores[-50:]) / len(agent.scores[-50:]))
    if str(episode)[-2:] == "00":# much faster than episode % 100
        pylab.plot(agent.episodes, agent.scores, 'b')
        pylab.plot(agent.episodes, agent.average, 'r')
        pylab.title(agent.agent_name + agent.env_name, fontsize=18 )
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Episodes', fontsize=18)
        try:
            pylab.savefig(agent.Plot_name+".png")
        except OSError:
            pass
    return agent.average[-1]