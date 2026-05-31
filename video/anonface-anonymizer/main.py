"""Main entry point - redirects to cli module."""

from cli import main

if __name__ == "__main__":
    main()
# python /Users/alyssa/Desktop/llm_as_a_judge/data/llm/video/anonface-anonymizer/main.py \
#   --input "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/Zoom Class Meeting Downing Soc 220 2⧸4⧸2021/images"\
#   --output "/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/Zoom Class Meeting Downing Soc 220 2⧸4⧸2021/images_black" \
#   --mode black \
#   --copy-on-fail