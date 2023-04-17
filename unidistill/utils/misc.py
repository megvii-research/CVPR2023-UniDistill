import re
import unicodedata


def sanitize_filename(value, allow_unicode=False):
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def parse_devices(gpu_ids):
    if "-" in gpu_ids:
        gpus = gpu_ids.split("-")
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        parsed_ids = ",".join(map(lambda x: str(x), list(range(*gpus))))
        return parsed_ids
    else:
        return gpu_ids


class PyDecorator:
    @staticmethod
    def overrides(interface_class):
        def overrider(method):
            assert method.__name__ in dir(
                interface_class
            ), "{} function not in {}".format(method.__name__, interface_class)

            return method

        return overrider
