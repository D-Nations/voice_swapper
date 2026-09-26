# Soundboard staging

One Markdown file per tab of the Very Bad Robots soundboard. Edit these, then run:

```
uv run python -m voice_service.export_soundboard
```

That encodes the clips into `soundboard/`, rewrites `soundboard/clips.json` and builds the site into
`soundboard/_site/`. Nothing is published until the soundboard repo is pushed.

## Files

- Name each file `NN-<tab id>.md`. The numbers set the tab order, and the tab id becomes the page's name
  (`02-tam-9000.md` becomes `tam-9000.html`).
- The first file is the home page. Its `#` heading is the site's title, and its tab is called Home.
- This README isn't a tab, since its name doesn't start with a number.

## Inside a file

```markdown
# TAM-9000

HAL 9000's lines from *2001: A Space Odyssey* (1968), in TamBot's voice.

## "Open the pod bay doors, please."

- audio: data/rvc/samples/taglines_renamed/takes/tam9000_podbay_e220_i0.4.wav
- voices: TamBot
- id: podbay
The clean take.
```

- `#` names the tab. Text before the first clip is the tab's intro.
- Each `##` is a clip, labeled as it appears on the page.
- `audio:` is the WAV to use, relative to the voice_swapper folder. `voices:` is who is heard: DaveBot, TamBot,
  or both separated by a comma. `id:` is optional and names the MP3; otherwise it's made from the label.
- Any other lines under a clip are a short note shown with it.
- Intros and notes can use paragraphs, `-` bullet lists, `###` subheadings, `**bold**`, `*italics*`, `` `code` ``
  and `[links](https://...)`. Links must start with `https://` or be relative.
- To drop a clip without deleting it, wrap it in `<!-- -->`.

The "Every voice here is AI-generated" notice is part of every page's template, not these files.
