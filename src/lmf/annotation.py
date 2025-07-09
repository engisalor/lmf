from pathlib import Path

import click
import pandas as pd


def bump_entities(row, by):
    ls = []
    if by == 0:
        size = +57
    else:
        size = 57 + ((len(row["text"]) + 57) * by)
    for dt in row["entities"]:
        ls.append(
            dt
            | dict(
                start_offset=dt["start_offset"] + size,
                end_offset=dt["end_offset"] + size,
                # label=row["annotator"],
            )
        )
    return ls


def validate_relation(row):
    ls = []
    for relation in row["relations"]:
        from_id = relation["from_id"]
        to_id = relation["to_id"]
        from_entity = [x for x in row["entities"] if x["id"] == from_id][0]
        to_entity = [x for x in row["entities"] if x["id"] == to_id][0]
        from_id_start_offset = from_entity["start_offset"]
        from_id_end_offset = from_entity["end_offset"]
        to_id_start_offset = to_entity["start_offset"]
        to_id_end_offset = to_entity["end_offset"]

        from_id_index = pd.Interval(from_id_start_offset, from_id_end_offset)
        to_id_index = pd.Interval(to_id_start_offset, to_id_end_offset)

        if not from_id_index.overlaps(to_id_index):
            ls.append(relation)
        else:
            click.echo(
                f"... invalid relation - id {relation['id']} - {row['annotator']}"
            )
    return ls


@click.command(
    context_settings={"show_default": True},
)
@click.option(
    "-v",
    "--verbose/--no-verbose",
    default=True,
    help="Increase verbosity",
)
@click.pass_context
def combine(ctx, verbose):
    """Combines annotation files in ./output for comparison in Docanno."""
    # explicit ctx
    project_dir: Path = ctx.obj["project_dir"]
    output_dir: Path = ctx.obj["output_dir"]
    debug: str = ctx.obj["debug"]

    click.echo(
        "... combining annotation docs - to view, load 'annotations-combined.jsonl' in Doccano"
    )

    files = output_dir.glob("*.jsonl")
    files = [x for x in files]
    output = project_dir / Path("annotations-combined.jsonl")
    annotators = ["gold"] + sorted(
        [x.with_suffix("").name for x in files if not x.stem.startswith("gold")]
    )
    click.echo(f"... annotators: {annotators}")

    def sorter(column: pd.Series):
        dt = {annotator: order for order, annotator in enumerate(annotators)}
        return column.map(dt)

    def copy_text(text):
        return "".join(
            [f"\n{i:02} : {a.ljust(50)}\n{text}" for i, a in enumerate(annotators)]
        )

    dfs = [pd.read_json(file, lines=True) for file in files]

    for file, df in zip(files, dfs):
        df["annotator"] = file.with_suffix("").name

    df = pd.concat(dfs)
    df = df.sort_values("annotator", key=sorter)

    for by, name in enumerate(annotators):
        df.loc[df["annotator"] == name, "entities"] = df.loc[
            df["annotator"] == name
        ].apply(bump_entities, by=by, axis=1)

    if verbose:
        _ = df.apply(validate_relation, axis=1)

    groups = [x[1] for x in df.groupby("id")]

    ls = []
    for group in groups:
        ls.append(
            dict(
                id=group.iloc[0]["id"],
                text=copy_text(group.iloc[0]["text"]),
                entities=[y for x in group["entities"].tolist() for y in x],
                relations=[y for x in group["relations"].tolist() for y in x],
                Comments=[y for x in group["Comments"].tolist() for y in x],
            )
        )

    final = pd.DataFrame.from_records(ls)

    if debug:
        for x in range(len(final)):
            text = final.iloc[x]["text"]
            entities = final.iloc[x]["entities"]
            for i, e in enumerate(entities):
                msg = (
                    f"text {x + 1}, entity {i + 1} "
                    + text[e["start_offset"] : e["end_offset"]]
                )
                click.echo(msg)

    final.to_json(output, orient="records", lines=True, force_ascii=False)
