# Custom Prompt Organization

AI File Butler now supports custom prompts for LLM-powered file organization! This allows you to specify exactly how you want your files organized based on your specific needs and preferences.

## Overview

The custom prompt feature lets you provide natural language instructions to the LLM about how to organize your files. Instead of using predefined rules, you can tell the AI exactly what criteria to use for organization.

## Usage

### CLI Command

Use the `organize-with-prompt` command to organize files with a custom prompt:

```bash
file-butler organize-with-prompt /path/to/files --prompt "Organize by year"
```

### Options

- `--prompt, -p`: **Required**. Your custom organization prompt
- `--provider`: LLM provider (openai, anthropic, local) - default: openai
- `--model`: LLM model to use - default: gpt-3.5-turbo
- `--cost-limit`: Maximum cost limit in USD - default: 5.0
- `--execute`: Execute the plan (default is dry run)
- `--verbose, -v`: Verbose output

### Examples

#### Organize by Year
```bash
file-butler organize-with-prompt /path/to/photos --prompt "Organize these files by the year they were created or modified. Create folders like '2024', '2023', '2022', etc."
```

#### Organize by Location
```bash
file-butler organize-with-prompt /path/to/travel_photos --prompt "Organize these files by geographic location. Extract location information from file names, metadata, or content. Create folders like 'Paris', 'Tokyo', 'New York', etc."
```

#### Organize by Project
```bash
file-butler organize-with-prompt /path/to/work_files --prompt "Organize these files by project name. Look for project identifiers in file names or content. Create folders for each unique project."
```

#### Organize by Event
```bash
file-butler organize-with-prompt /path/to/event_photos --prompt "Organize these files by event or occasion. Look for event names, dates, or themes in the content. Create folders like 'Wedding', 'Birthday Party', 'Conference', etc."
```

## Prompt Examples

### By Year
```
Organize these files by the year they were created or modified. Create folders like '2024', '2023', '2022', etc. Look for dates in file names, metadata, or content.
```

### By Location
```
Organize these files by geographic location. Extract location information from file names, metadata, or content. Create folders like 'New York', 'London', 'Tokyo', etc.
```

### By Client
```
Organize these files by client name. Extract client information from file names, email addresses, or document content. Create a folder for each client.
```

### By Priority
```
Organize these files by priority level. Create folders like 'High Priority', 'Medium Priority', 'Low Priority', 'Archive' based on importance and urgency.
```

### By Topic
```
Organize these files by academic or professional topic. Create folders for different subjects like 'Marketing', 'Finance', 'Technology', 'Health', etc.
```

### By File Type and Purpose
```
Organize these files by their primary purpose and type. Create folders like 'Documents/Contracts', 'Images/Logos', 'Videos/Tutorials', 'Audio/Podcasts'.
```

## Tips for Writing Effective Prompts

1. **Be Specific**: Clearly state what information to extract (dates, names, locations, etc.)
2. **Mention Folder Structure**: Specify the desired folder naming convention
3. **Include Examples**: Provide examples of expected folder names
4. **Consider Context**: Think about what information is available in your files
5. **Test First**: Always run in dry-run mode first to see the results

## Viewing Examples

To see more prompt examples, run:

```bash
file-butler prompt-examples
```

This will show you a comprehensive list of example prompts with their use cases and command syntax.

## Integration with Other Features

Custom prompts work seamlessly with other AI File Butler features:

- **Cost Tracking**: Monitor your LLM usage costs
- **Dry Run Mode**: Test your prompts without making changes
- **Verbose Output**: See detailed analysis and reasoning
- **Caching**: Avoid re-analyzing the same files

## Advanced Usage

### Using with Regular Organize Command

You can also use custom prompts with the regular `organize` command:

```bash
file-butler organize /path/to/files --strategy llm --custom-prompt "Organize by year"
```

### Programmatic Usage

```python
from file_butler.engines.llm import LLMOrganizationEngine, LLMConfig

config = LLMConfig(
    provider='openai',
    model='gpt-3.5-turbo',
    custom_prompt="Organize these files by project name"
)

engine = LLMOrganizationEngine(config)
plan = engine.organize(files)
```

## Cost Considerations

Custom prompts use the same cost tracking as regular LLM organization. The cost depends on:

- Number of files analyzed
- File content length
- Complexity of the prompt
- LLM model used

Set a cost limit to avoid unexpected charges:

```bash
file-butler organize-with-prompt /path/to/files --prompt "your prompt" --cost-limit 2.0
```

## Troubleshooting

### Common Issues

1. **API Key Not Set**: Make sure `OPENAI_API_KEY` is set
2. **Cost Limit Reached**: Increase the cost limit or reduce file count
3. **Unclear Results**: Make your prompt more specific
4. **No Files Found**: Check the source path

### Getting Help

- Run `file-butler prompt-examples` for examples
- Use `--verbose` flag for detailed output
- Test with a small number of files first
- Check the cost tracking to monitor usage

## Best Practices

1. **Start Simple**: Begin with basic prompts and refine them
2. **Test Thoroughly**: Always use dry-run mode first
3. **Monitor Costs**: Keep track of your LLM usage
4. **Backup Data**: Ensure you have backups before executing changes
5. **Iterate**: Refine your prompts based on results

The custom prompt feature gives you unprecedented control over how your files are organized, allowing you to create exactly the structure you need for your specific use case. 