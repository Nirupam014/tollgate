from crewai import Agent, Task, Crew

a = Agent(role="Classifier", goal="x", backstory="b", llm="gpt-4o-mini")
t = Task(description="Classify the ticket.", agent=a)
crew = Crew(agents=[a], tasks=[t])
crew.kickoff()
