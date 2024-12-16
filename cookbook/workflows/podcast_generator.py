from typing import List
from pydantic import BaseModel, Field
from phi.agent import Agent
from phi.model.openai import OpenAIChat
import dotenv
import json
from pathlib import Path
from datetime import datetime
import sys
from phi.utils.log import logger

# Add the project root directory to Python path
project_root = Path(__file__).resolve().parents[2]  # Adjust the path as needed
sys.path.append(str(project_root))

# Now we can import from tools after adding to path
from tools.content_generation_toolkit import ContentGenerationToolkit
from tools.text_to_voice_toolkit import TextToVoiceTools

"""
Sourced from https://github.com/ayyzad/phidata-agent-examples/blob/main/agents/create-podcast-agent/podcast_agent.py
"""

class DialogueMessage(BaseModel):
    speaker: str
    message: str


class PodcastDialogue(BaseModel):
    messages: List[DialogueMessage] = Field(
        ...,
        description="A list of messages alternating between Person A and Person B discussing and summarizing the content. Make it sound casual and engaging while covering the main points.",
    )


# Agent that uses structured outputs
dialogue_agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    description="You create natural dialogue summaries between two people discussing content. Alternate between Person A and Person B.",
    response_model=PodcastDialogue,
    structured_outputs=True,
)


def create_podcast_dialogue(topic: str = "Latest developments in AI"):
    """Create podcast dialogue from generated content on a topic"""
    from phi.storage.workflow.sqlite import SqlWorkflowStorage

    logger.info(f"Starting podcast dialogue creation for topic: '{topic}'")

    # Initialize the content generation toolkit
    content_tools = ContentGenerationToolkit()
    logger.debug("Initialized ContentGenerationToolkit")

    content_tools.initialize_workflow(
        session_id="podcast_generation_session",
        storage=SqlWorkflowStorage(
            table_name="content_generation_workflows",
            db_file="tmp/workflows.db",
        ),
    )
    logger.debug("Workflow initialized with SqlWorkflowStorage")

    # Generate blog post using the toolkit
    logger.info("Generating blog post content")
    blog_post_generator = content_tools.generate_blog_post(topic)

    # Process the generator to get the blog post content
    blog_post_content = None
    for response in blog_post_generator:
        logger.debug(f"Received response: {response.content}")
        if response.content.startswith("Sorry"):
            logger.error(f"Failed to generate blog post: {response.content}")
            print(response.content)
            return
        blog_post_content = response.content

    if not blog_post_content:
        logger.error(f"Failed to generate blog post for topic: {topic}")
        return

    logger.info(f"Generated blog post content: {blog_post_content[:100]}...")  # Log the first 100 characters

    # Save the blog post content to a file
    base_output_dir = Path(__file__).parent
    blog_post_file = base_output_dir / "podcast-agent-output" / "blog-posts" / f"{topic.replace(' ', '_')}_blog_post.md"
    blog_post_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(blog_post_file, "w", encoding="utf-8") as file:
            file.write(blog_post_content)
        logger.info(f"Blog post saved successfully to: {blog_post_file}")

        # Verify the file was written
        if not blog_post_file.exists() or blog_post_file.stat().st_size == 0:
            logger.error("Blog post file is empty or wasn't created")
            return
    except Exception as e:
        logger.error(f"Failed to save blog post: {str(e)}")
        raise

    # Get the dialogue response
    logger.info("Generating dialogue from blog post")
    try:
        dialogue_response = dialogue_agent.run(blog_post_content)
        logger.info(f"Generated dialogue response: {dialogue_response.content}")
        logger.info("Successfully generated dialogue")
    except Exception as e:
        logger.error(f"Failed to generate dialogue: {str(e)}")
        raise

    # Create output directory
    output_dir = base_output_dir / "podcast-agent-output" / "conversation-transcriptions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{topic.replace(' ', '_')}_dialogue.json"

    # Save dialogue
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dialogue": dialogue_response.content.dict(),
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "topic": topic,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"Dialogue saved successfully to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save dialogue: {str(e)}")
        raise

    # Create audio
    logger.info("Starting audio generation")
    try:
        voice_tools = TextToVoiceTools()
        audio_result = voice_tools.create_conversation_audio(str(output_file))
        logger.info("Successfully generated audio")
        return audio_result
    except Exception as e:
        logger.error(f"Failed to generate audio: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage with a specific topic
    create_podcast_dialogue("current events of palantir company")
