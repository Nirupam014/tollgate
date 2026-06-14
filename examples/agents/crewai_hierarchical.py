from crewai import Agent, Task, Crew, Process

researcher = Agent(role="Researcher", goal="dig", backstory="b",
                   llm="gpt-4o", allow_delegation=True)
writer = Agent(role="Writer", goal="write", backstory="b", llm="gpt-4o-mini")

t1 = Task(description="Research the competitive landscape in depth.", agent=researcher)
t2 = Task(description="Write a long report from the research.", agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[t1, t2],
            process=Process.hierarchical, memory=True, manager_llm="gpt-4o")
result = crew.kickoff()
