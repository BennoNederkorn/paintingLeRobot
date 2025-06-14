import numpy as np

def execute_recording() -> list[float]:
    """
    This function strarts a recording while a key is pressed and returns a .wav file/data
    """
    return []


def speech_to_text(data : list[float]) -> str:
    """
    This function gets a speach data and returns text
    """
    
    new_prompt : str = "Hello Lee" 
    print(new_prompt)
    return new_prompt

def generate_image(prompt : str) -> np.ndarray:
    return np.random.rand(255, 255, 3)

def create_segmentated_image(image : np.ndarray) -> np.ndarray:
    return np.random.rand(255, 255, 3)


def main():
    has_robot_finished_image = True
    while(has_robot_finished_image):
        new_speach_prompt : list[float] = execute_recording()
        new_text_prompt : str = speech_to_text(new_speach_prompt)
        image : np.ndarray = generate_image(new_text_prompt)
        segmentated_image : np.ndarray = create_segmentated_image(image)




if __name__ == "__main__":
    main()